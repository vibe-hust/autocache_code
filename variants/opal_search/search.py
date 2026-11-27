"""Independent OPAL search: Pareto feedback, then final development representatives."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = 'variants.opal_search'

from reactive_samplers.data import read_prompts
from reactive_samplers.report import write_json
from .agent import round_input, invoke, safe_child, strategy_entry
from .diagnostics import diagnose
from .evaluate import Evaluator, read, validate_rows, valid_result
from .parallel import make_evaluator, evaluate_jobs
from .policy import freeze, source_hash, structure_hash, Runtime
from .selection import configure, DEFAULT_TARGETS, FLUX_TARGETS, METRICS, LOSS_FORMULA, QUALITY_GATE, REPRESENTATIVE_METRICS, quality_report, pareto, select_representatives, paired


def read_rows(path):
    count = sum(bool(line.strip()) for line in Path(path).read_text(encoding='utf-8').splitlines())
    return read_prompts(path, count)


def validate_split(development, test):
    for key in ('id', 'prompt'):
        if {row[key] for row in development} & {row[key] for row in test}:
            raise ValueError('Final test prompts must be disjoint from development prompts')


def configure_targets(config, generation):
    defaults = FLUX_TARGETS if generation['model'] == 'flux3' else DEFAULT_TARGETS
    configure(config.get('quality_targets', defaults))


def reference_directory(output):
    output = Path(output).resolve()
    name = 'references' if output.name == 'search' else f'{output.name}-references'
    return output.with_name(name)


def load_saved_config(path):
    output = Path(read(path)['output_dir'])
    config = read(output/'inputs.json')['config']
    configure_targets(config, config['generation'])
    return config


def load_config(path):
    config=read(path)
    generation=read(config['generation_config'])
    if generation.get('model') not in ('qwen21', 'flux3', 'flux') or generation.get('method')!='residual_cache':
        raise ValueError('OPAL supports Qwen21, FLUX3 and FLUX.1 residual_cache')
    if generation.get('reference_latents') or generation.get('record_reference_latents'):
        raise ValueError('OPAL does not consume or record official latent trajectories')
    if not set(METRICS)<=set(generation['metrics']['names']): raise ValueError('All five metrics required')
    needed=('model','method','model_path','dtype','device','height','width','guidance_scale','max_sequence_length',
            'reference_steps','metrics_python','metrics')
    sampling_profile=generation.get('sampling_profile')
    generation={k:generation[k] for k in needed}
    if sampling_profile is not None: generation['sampling_profile']=sampling_profile
    if type(generation['reference_steps']) is not int or generation['reference_steps']<1: raise ValueError('Positive steps required')
    if config.get('observation_fields') is not None: generation['observation_fields'] = config['observation_fields']
    configure_targets(config, generation)
    config={**config,'generation':generation}
    config.setdefault('round_sizes',[10]+[5]*8)
    config.setdefault('batch_size',1)
    config.setdefault('diagnostic_budget',9)
    config.setdefault('max_agent_calls_per_round',12)
    config.setdefault('max_replacement_rounds',3)
    config.setdefault('state_tensor_bytes',64*1024*1024)
    config.setdefault('diagnostic_tensor_bytes',256*1024*1024)
    for key in ('batch_size','max_agent_calls_per_round','state_tensor_bytes','diagnostic_tensor_bytes'):
        if type(config[key]) is not int or config[key]<1: raise ValueError('Positive integer required: '+key)
    if type(config['diagnostic_budget']) is not int or config['diagnostic_budget']<0: raise ValueError('Invalid diagnostic budget')
    if type(config['max_replacement_rounds']) is not int or config['max_replacement_rounds']<0: raise ValueError('Invalid replacement budget')
    if not config['round_sizes'] or any(type(x) is not int or x<1 for x in config['round_sizes']) or sum(config['round_sizes'])<3:
        raise ValueError('Round sizes must be positive, with at least three strategies')
    rows=read_rows(config['development_prompts']); validate_rows(rows)
    test_rows=read_rows(config['test_prompts'])
    validate_split(rows, test_rows)
    if not config['agent'].get('command') and not config['agent'].get('model'): raise ValueError('Explicit Agent model required')
    config['rows']=rows
    config['test_rows']=test_rows
    config['loss']={'formula':LOSS_FORMULA,'psnr_null':'positive infinity; image loss=0',
                    'initial_loss':1,'normalization':False,'aggregation':'mean valid per-image loss; nonpositive denominator=null'}
    config['quality_gate']=json.loads(json.dumps(QUALITY_GATE))
    config['output_dir']=str(Path(config['output_dir']).resolve())
    reference_keys = ('reference_results', 'test_reference_results')
    prepared_path = reference_directory(config['output_dir'])/'search.json'
    if prepared_path.is_file() and not any(key in config for key in reference_keys):
        prepared = read(prepared_path)
        config.update({key: prepared[key] for key in reference_keys})
    evaluator = Evaluator(generation, config['batch_size'], config['state_tensor_bytes'])
    for key, split in zip(reference_keys, (rows, test_rows)):
        if config.get(key):
            valid_result(config[key], evaluator.context(split, None, True, None))
    return config


def save(root, state):
    for record in state['records']:
        if record['status']=='completed': record['quality_gate']=quality_report(record['summary'])
    state['frontier']=pareto(state['records'])
    write_json(root/'state.json',state)
    history = [strategy_entry(record, state['frontier']) for record in state['records']
               if record['status'] == 'completed' and record['id'] not in state['frontier']
               and not (record['id'] != 'initial' and quality_report(record['summary'])['eligible'])]
    write_json(root/'leaderboard.json', {'strategies': history})
    usages=[read(p) for p in (root/'rounds').glob('*/call_*/usage.json')]
    write_json(root/'usage.json',{k:sum(u.get(k,0) for u in usages) for k in
        ('input_tokens','output_tokens','calls_with_usage','calls_without_usage','seconds')})


def record_result(root, record, data, state):
    record.update(status='completed',summary=data['summary'])
    parent=next(r for r in state['records'] if r['id']==record['parent'])
    reference=next(r for r in state['records'] if r['id']=='initial')
    feedback={'id':record['id'],'parent':record['parent'],
              'versus_parent':paired(data['images'],valid_result(parent['results'])['images'])}
    if parent['id']!='initial': feedback['versus_initial']=paired(data['images'],valid_result(reference['results'])['images'])
    record['feedback']=feedback
    save(root,state)


def run(config, evaluator=None, agent_call=invoke, resume=False):
    root=Path(config['output_dir'])
    provenance={'config':config}
    if resume:
        recorded=read(root/'inputs.json')
        if recorded['config']!=config: raise ValueError('Resume requires identical experiment configuration and data')
        state=read(root/'state.json')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/'inputs.json',provenance)
        policy=freeze(Path(__file__).parent/'templates',root/'strategies/initial',config['generation']['model'])
        state={'records':[{'id':'initial','round':0,'status':'pending','policy':str(policy),
            'sha256':source_hash(policy),'structure_sha256':structure_hash(policy),
            'results':str(root/'evaluations/initial/results.json')}], 'frontier':[], 'diagnostics_used':0}
        save(root,state)
    runner=evaluator or make_evaluator(config)
    try:
        initial=state['records'][0]
        if initial['status']!='completed':
            write_json(root/'status.json',{'status':'running','stage':'reference'})
            if config.get('reference_results'):
                data=valid_result(config['reference_results'],runner.context(config['rows'],initial['policy'],True,None))
                initial['results']=str(Path(config['reference_results']).resolve())
            else:
                data=runner.run(config['rows'],initial['policy'],Path(initial['results']).parent,initial=True)
            if any(r['nfe']!=config['generation']['reference_steps']*(2 if config['generation']['model']=='flux3' else 1) for r in data['images']): raise ValueError('Initial reference must be all NEW')
            initial.update(status='completed',summary=data['summary']);save(root,state)
        for number,target in enumerate(config['round_sizes'],1):
            if str(number) in state.get('round_outcomes',{}): continue
            pending=[r for r in state['records'] if r['round']==number and r['status']=='pending']
            while True:
                # Evaluate a whole submitted batch before requesting the next Agent action.
                jobs=[{'id':r['id'],'policy':r['policy'],'output':str(Path(r['results']).parent)} for r in pending]
                if jobs:
                    write_json(root/'status.json',{'status':'running','round':number,'stage':'evaluation',
                        'strategies':[j['id'] for j in jobs], 'candidate_gpus':config.get('evaluation_gpus', [0, 1])})
                for job, measured, error in evaluate_jobs(runner,jobs,config['rows'],initial['results']):
                    record=next(r for r in pending if r['id']==job['id'])
                    if error is None:
                        record_result(root,record,measured,state)
                    else:
                        record.update(status='failed',error=error)
                        record['feedback']={'id':record['id'],'error':record['error']}
                        save(root,state)
                pending=[]
                completed=sum(r['round']==number and r['status']=='completed' for r in state['records'])
                if completed>target: raise ValueError('Round success budget exceeded')
                folder=root/'rounds'/f'round_{number:03d}';folder.mkdir(parents=True,exist_ok=True)
                calls=sorted(folder.glob('call_*'))
                submissions=sum((call/'submission.json').exists() for call in calls)
                reason='slots_filled' if completed==target else 'replacement_budget_exhausted' if submissions>=1+config['max_replacement_rounds'] else None
                if reason:
                    state.setdefault('round_outcomes',{})[str(number)]={'reason':reason,
                        'successful':completed,'initial_slots':target,'submission_batches':submissions}
                    save(root,state)
                    break
                counted_calls=sum(not ((call/'feedback.json').exists()
                    and 'diagnostic' in read(call/'feedback.json')) for call in calls)
                if counted_calls>=config['max_agent_calls_per_round']:
                    raise RuntimeError('Agent call budget exhausted; search incomplete')
                workspace=folder/f'call_{len(calls)+1:03d}';workspace.mkdir()
                feedback=[r['feedback'] for r in state['records'] if r.get('feedback') and r['round'] in (number-1,number)]
                if calls and (calls[-1]/'feedback.json').exists(): feedback.append(read(calls[-1]/'feedback.json'))
                round_input(root,workspace,number,target-completed,state,config,feedback)
                write_json(root/'status.json',{'status':'running','round':number,'stage':'agent','call':len(calls)+1})
                kind,payload=agent_call(root,workspace,config)
                try:
                    if kind=='diagnostic_request.json':
                        if payload['kind']=='probe':
                            if state['diagnostics_used']>=config['diagnostic_budget']: raise ValueError('Probe budget exhausted')
                            state['diagnostics_used']+=1;save(root,state)
                        answer=diagnose(payload,workspace,root/'diagnostics'/f'{folder.name}_{workspace.name}',
                            state['records'],config['rows'],runner,initial['results'],config)
                        write_json(workspace/'feedback.json',{'diagnostic':answer});continue
                    items=payload['strategies']
                    if not isinstance(items,list) or len(items)!=target-completed:
                        raise ValueError('Submit exactly the requested number of new packages')
                    known={r['structure_sha256'] for r in state['records'] if r['status'] in ('pending','completed')}
                    ids={r['id'] for r in state['records']};validated=[]
                    for item in items:
                        sid=item['id']
                        if not isinstance(sid,str) or not sid or len(sid)>80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid) or sid in ids:
                            raise ValueError('Unique safe strategy ID required')
                        parent=next((r for r in state['records'] if r['id']==item['parent'] and r['status']=='completed'),None)
                        if parent is None: raise ValueError('Parent must name a completed reference strategy')
                        for key in ('change','evidence','hypothesis'):
                            if not isinstance(item.get(key),str) or not item[key].strip() or len(item[key])>2000:
                                raise ValueError('Short nonempty '+key+' required')
                        source=safe_child(workspace,item['path'])
                        if config.get('fixed_signals') and (source/'signals.py').read_text() != (Path(__file__).parent/'templates/signals.py').read_text():
                            raise ValueError('signals.py is fixed in this ablation; edit only updater.py')
                        identity=structure_hash(source)
                        if identity in known: raise ValueError('Duplicate strategy AST: '+sid)
                        known.add(identity);ids.add(sid);validated.append((item,source))
                    for item,source in validated:
                        policy=root/'strategies'/item['id']
                        if policy.exists():
                            if source_hash(policy)!=source_hash(source): raise ValueError('An interrupted package uses this ID; use a new ID')
                        else:
                            freeze(source,policy,config['generation']['model'])
                        # Load both functions without executing a model or an Agent report.
                        Runtime(policy,config['state_tensor_bytes'])
                        record={**{k:item[k] for k in ('id','parent','change','evidence','hypothesis')},
                            'round':number,'status':'pending','policy':str(policy),'sha256':source_hash(policy),
                            'structure_sha256':structure_hash(policy),'results':str(root/'evaluations'/item['id']/'results.json')}
                        state['records'].append(record);pending.append(record);save(root,state)
                except (ValueError,KeyError,TypeError,SyntaxError,FileNotFoundError) as exc:
                    write_json(workspace/'feedback.json',{'submission_error':f'{type(exc).__name__}: {str(exc)[-2000:]}'})
                    continue
            save(root,state)
        for record in state['records']:
            if record['status']=='completed': valid_result(record['results'])
        representatives=select_representatives(state['records'])
        chosen=list({r['id']:r for r in representatives.values()}.values())
        assignments={category:{'id':r['id'],'metric':REPRESENTATIVE_METRICS[category][0],
                     'development_value':r['summary'][REPRESENTATIVE_METRICS[category][0]],
                     'quality_eligible':quality_report(r['summary'])['eligible']}
                     for category,r in representatives.items()}
        qualified=sum(r['status']=='completed' and r['id']!='initial' and quality_report(r['summary'])['eligible'] for r in state['records'])
        selection_status=('selected' if qualified else 'fallback_no_eligible_new_strategy') if chosen else 'no_valid_new_strategy'
        write_json(root/'quality_report.json',{'quality_requirement':QUALITY_GATE,'qualified_new_strategies':qualified,
            'selection_status':selection_status,'selected_ids':[r['id'] for r in chosen],
            'pareto':[{'id':r['id'],'summary':r['summary'],'quality_gate':r['quality_gate'],'policy':r['policy']}
                      for r in state['records'] if r['id'] in state['frontier']]})
        write_json(root/'selected_representatives.json',{'selection':'After all search rounds, use qualifying new strategies if any, otherwise all completed new strategies. Independently select highest CLIPScore, highest ImageReward, lowest LPIPS, lowest NFE and lowest loss using development means; ties use NFE, time, then ID. Categories may share a strategy. Initial excluded. Test each unique strategy once.',
            'selection_status':selection_status,'quality_requirement':QUALITY_GATE,
            'representatives':assignments,
            'strategies':[{'id':r['id'],'policy':r['policy'],'sha256':r['sha256'],'summary':r['summary'],
                           'categories':[name for name,value in representatives.items() if value['id']==r['id']],
                           'quality_eligible':quality_report(r['summary'])['eligible']}
                          for r in chosen]})
        write_json(root/'status.json',{'status':'completed','successful_new_strategies':sum(r['id']!='initial' and r['status']=='completed' for r in state['records']),
            'quality_eligible_strategies':qualified,'selected_count':len(chosen),
            'representative_count':len(assignments),'selection_status':selection_status})
        return state
    except BaseException as exc:
        save(root,state)
        write_json(root/'status.json',{'status':'incomplete','error':f'{type(exc).__name__}: {exc}'})
        raise
    finally: runner.close()


def final_table(reference_summary, strategies):
    lines = ['| Representative(s) | Strategy | CLIPScore ↑ | ImageReward ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NFE ↓ | Seconds/image ↓ |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in [{'id':'initial', 'summary':reference_summary}, *strategies]:
        s=row['summary']
        values=[('∞' if s[k] is None else f'{s[k]:.4f}') for k in
                ('mean_clip_score','mean_image_reward','mean_psnr','mean_ssim','mean_lpips','mean_nfe','mean_seconds')]
        categories=', '.join(row.get('categories', ['reference']))
        lines.append('| ' + categories + ' | ' + row['id'] + ' | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def test_selected(config):
    root=Path(config['output_dir']);selection=read(root/'selected_representatives.json')
    rows=config['test_rows'] if 'test_rows' in config else read_rows(config['test_prompts'])
    validate_rows(rows)
    validate_split(config['rows'], rows)
    if not selection['strategies']:
        write_json(root/'test/results.json',{'strategies':[],'representatives':{},'selection_status':'no_valid_new_strategy',
            'note':'No completed new strategy available. No test generation or automatic budget extension.',
            'quality_report':str(root/'quality_report.json')})
        (root/'test/metrics.md').write_text('No completed new strategy available.\n', encoding='utf-8')
        return
    runner=make_evaluator(config)
    try:
        reference=Path(config.get('test_reference_results',root/'test/reference/results.json'))
        if config.get('test_reference_results'):
            ref=valid_result(reference,runner.context(rows,root/'strategies/initial',True,None))
            write_json(root/'test/reference/results.json', ref)
        else:
            ref=runner.run(rows,root/'strategies/initial',root/'test/reference',initial=True)
        shared_reference = reference_directory(root)/'test/results.json'
        if reference.resolve() != shared_reference:
            write_json(shared_reference, ref)
        results=[]
        jobs=[]
        for r in selection['strategies']:
            jobs.append({'id':r['id'],'policy':r['policy'],'output':str(root/'test'/r['id'])})
        for job, measured, error in evaluate_jobs(runner,jobs,rows,reference):
            if error is not None: raise RuntimeError(error)
            r=next(r for r in selection['strategies'] if r['id']==job['id'])
            results.append({'id':r['id'],'categories':r['categories'],'summary':measured['summary'],
                            'results':str(root/'test'/r['id']/'results.json')})
        order={r['id']:i for i,r in enumerate(selection['strategies'])}
        results.sort(key=lambda r:order[r['id']])
        write_json(root/'test/results.json',{'selection_status':selection['selection_status'],
            'reference_results':str(reference),
            'reference_summary':ref['summary'],'strategies':results,'representatives':selection['representatives'],
            'selection':'Categories fixed on development data after all search rounds; unique strategies tested once. Test scores do not reselect representatives or feed the Agent.'})
        (root/'test/metrics.md').write_text(final_table(ref['summary'], results), encoding='utf-8')
    finally: runner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','test'])
    parser.add_argument('config');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    use_saved = args.command=='test' or (args.command=='run' and args.resume)
    config=load_saved_config(args.config) if use_saved else load_config(args.config)
    if args.command=='check':
        print(json.dumps({'valid':True,'images':len(config['rows']),'round_sizes':config['round_sizes'],
                          'loss':config['loss'],'gpu_generation':False,'paid_agent_calls':False}))
    elif args.command=='run':
        controller = run
        if config.get('feedback_mode') == 'deferred':
            from .deferred_search import run as controller
        controller(config,resume=args.resume)
    else: test_selected(config)


if __name__=='__main__':main()
