"""Independent OPAL search: Pareto feedback, then final development representatives."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = 'variants.wan21_opal'

from reactive_samplers.report import write_json
from .agent import round_input, invoke, safe_child, strategy_entry
from .diagnostics import diagnose
from .evaluate import Evaluator, read, validate_rows, valid_result, fingerprint
from .parallel import make_evaluator, evaluate_jobs
from .policy import freeze, source_hash, structure_hash, Runtime
from .selection import METRICS, LOSS_FORMULA, QUALITY_GATE, REPRESENTATIVE_METRICS, quality_report, pareto, select_representatives, paired


def load_config(path):
    config=read(path)
    generation=read(config['generation_config'])
    if generation.get('model')!='wan21' or generation.get('method')!='residual_cache':
        raise ValueError('Initial OPAL integration supports Wan21 residual_cache')
    if generation.get('reference_latents') or generation.get('record_reference_latents'):
        raise ValueError('OPAL does not consume or record official latent trajectories')
    from .evaluate import validate_generation
    validate_generation(generation)
    config={**config,'generation':generation}
    config.setdefault('round_sizes',[10]+[5]*8)
    config.setdefault('batch_size',1)
    if config['batch_size'] != 1: raise ValueError('Wan video search currently uses batch_size=1')
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
    prompt_path=Path(config['development_prompts'])
    rows=[json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
    validate_rows(rows)
    if not config['agent'].get('command') and not config['agent'].get('model'): raise ValueError('Explicit Agent model required')
    config['rows']=rows
    config['loss']={'formula':LOSS_FORMULA,'psnr_null':'positive infinity; video loss=0',
                    'initial_loss':1,'normalization':False,'aggregation':'mean per-video loss'}
    config['quality_gate']=QUALITY_GATE
    config['output_dir']=str(Path(config['output_dir']).resolve())
    if config.get('reference_results'):
        expected=Evaluator(generation,config['batch_size'],config['state_tensor_bytes']).context(
            rows,Path(__file__).parent/'templates',True,None)
        valid_result(config['reference_results'],expected)
    return config


def save(root, state):
    for record in state['records']:
        if record['status']=='completed': record['quality_gate']=quality_report(record['summary'])
    state['frontier']=pareto(state['records'])
    write_json(root/'state.json',state)
    history = [strategy_entry(record, state['frontier']) for record in state['records']
               if record['status'] == 'early_stopped' or (record['status'] == 'completed'
               and record['id'] not in state['frontier']
               and not (record['id'] != 'initial' and quality_report(record['summary'])['eligible']))]
    write_json(root/'leaderboard.json', {'strategies': history})
    usages=[read(p) for p in (root/'rounds').glob('*/call_*/usage.json')]
    write_json(root/'usage.json',{k:sum(u.get(k,0) for u in usages) for k in
        ('input_tokens','output_tokens','calls_with_usage','calls_without_usage','seconds')})


def record_result(root, record, data, state):
    if data.get('screening', {}).get('status') == 'early_stopped':
        record.update(status='early_stopped', screening_summary=data['summary'],
                      results=str(Path(record['results']).parent/'screening.json'))
        record['feedback'] = {'id':record['id'], 'status':'early_stopped',
            'screening':data['screening'],
            'scores':{k:v for k,v in data['summary'].items() if k == 'count' or k.startswith('mean_')}}
        save(root,state)
        return
    record.update(status='completed',summary=data['summary'])
    parent=next(r for r in state['records'] if r['id']==record['parent'])
    reference=next(r for r in state['records'] if r['id']=='initial')
    parent_data = read(parent['results']) if parent['status']=='early_stopped' else valid_result(parent['results'])
    parent_ids = {r['id'] for r in parent_data['videos']}
    feedback={'id':record['id'],'parent':record['parent'],
              'versus_parent':paired([r for r in data['videos'] if r['id'] in parent_ids],parent_data['videos'])}
    if parent['status']=='early_stopped':
        feedback['parent_comparison_samples']=len(parent_ids)
    if parent['id']!='initial': feedback['versus_initial']=paired(data['videos'],valid_result(reference['results'])['videos'])
    record['feedback']=feedback
    save(root,state)


def run(config, evaluator=None, agent_call=invoke, resume=False):
    root=Path(config['output_dir'])
    provenance={'config':config}
    if resume:
        recorded=read(root/'inputs.json')
        if recorded['config']!=config: raise ValueError('Resume requires identical experiment configuration and data')
        # Older runs retain their recorded source hashes as historical metadata only.
        provenance=recorded
        state=read(root/'state.json')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/'inputs.json',provenance)
        policy=freeze(Path(__file__).parent/'templates',root/'strategies/initial')
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
            if any(r['nfe']!=2*config['generation']['reference_steps'] for r in data['videos']): raise ValueError('Initial reference must be all NEW')
            initial.update(status='completed',summary=data['summary']);save(root,state)
        for number,target in enumerate(config['round_sizes'],1):
            if str(number) in state.get('round_outcomes',{}): continue
            pending=[r for r in state['records'] if r['round']==number and r['status']=='pending']
            while True:
                # Evaluate a whole submitted batch before requesting the next Agent action.
                jobs=[{'id':r['id'],'policy':r['policy'],'output':str(Path(r['results']).parent),'screen':True} for r in pending]
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
                screened=sum(r['round']==number and r['status']=='early_stopped' for r in state['records'])
                evaluated=completed+screened
                if evaluated>target: raise ValueError('Round success budget exceeded')
                folder=root/'rounds'/f'round_{number:03d}';folder.mkdir(parents=True,exist_ok=True)
                calls=sorted(folder.glob('call_*'))
                submissions=sum((call/'submission.json').exists() for call in calls)
                reason='slots_filled' if evaluated==target else 'replacement_budget_exhausted' if submissions>=1+config['max_replacement_rounds'] else None
                if reason:
                    state.setdefault('round_outcomes',{})[str(number)]={'reason':reason,
                        'successful':completed,'early_stopped':screened,'initial_slots':target,'submission_batches':submissions}
                    save(root,state)
                    break
                counted_calls=sum(not ((call/'feedback.json').exists()
                    and 'diagnostic' in read(call/'feedback.json')) for call in calls)
                if counted_calls>=config['max_agent_calls_per_round']:
                    raise RuntimeError('Agent call budget exhausted; search incomplete')
                workspace=folder/f'call_{len(calls)+1:03d}';workspace.mkdir()
                feedback=[r['feedback'] for r in state['records'] if r.get('feedback') and r['round'] in (number-1,number)]
                if calls and (calls[-1]/'feedback.json').exists(): feedback.append(read(calls[-1]/'feedback.json'))
                round_input(root,workspace,number,target-evaluated,state,config,feedback)
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
                    if not isinstance(items,list) or len(items)!=target-evaluated:
                        raise ValueError('Submit exactly the requested number of new packages')
                    known={r['structure_sha256'] for r in state['records'] if r['status'] in ('pending','completed','early_stopped')}
                    ids={r['id'] for r in state['records']};validated=[]
                    for item in items:
                        sid=item['id']
                        if not isinstance(sid,str) or not sid or len(sid)>80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid) or sid in ids:
                            raise ValueError('Unique safe strategy ID required')
                        parent=next((r for r in state['records'] if r['id']==item['parent'] and r['status'] in ('completed','early_stopped')),None)
                        if parent is None: raise ValueError('Parent must name an evaluated reference strategy')
                        for key in ('change','evidence','hypothesis'):
                            if not isinstance(item.get(key),str) or not item[key].strip() or len(item[key])>2000:
                                raise ValueError('Short nonempty '+key+' required')
                        source=safe_child(workspace,item['path']);identity=structure_hash(source)
                        if identity in known: raise ValueError('Duplicate strategy AST: '+sid)
                        known.add(identity);ids.add(sid);validated.append((item,source))
                    for item,source in validated:
                        policy=root/'strategies'/item['id']
                        if policy.exists():
                            if source_hash(policy)!=source_hash(source): raise ValueError('An interrupted package uses this ID; use a new ID')
                        else:
                            freeze(source,policy)
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
        selection_status='selected' if qualified else 'fallback_no_eligible_new_strategy' if chosen else 'no_valid_new_strategy'
        write_json(root/'quality_report.json',{'quality_requirement':QUALITY_GATE,'qualified_new_strategies':qualified,
            'selection_status':selection_status,'selected_ids':[r['id'] for r in chosen],
            'pareto':[{'id':r['id'],'summary':r['summary'],'quality_gate':r['quality_gate'],'policy':r['policy']}
                      for r in state['records'] if r['id'] in state['frontier']]})
        write_json(root/'selected_representatives.json',{'selection':'After all search rounds, select each category independently among qualifying new strategies if any, otherwise among all completed new strategies and report no qualifying strategy, using development means; ties use NFE, time, then ID. Initial excluded. Test each unique strategy once.',
            'selection_status':selection_status,'quality_requirement':QUALITY_GATE,'qualified_new_strategies':qualified,
            'representatives':assignments,
            'strategies':[{'id':r['id'],'policy':r['policy'],'sha256':r['sha256'],'summary':r['summary'],
                           'quality_eligible':quality_report(r['summary'])['eligible'],
                           'categories':[name for name,value in representatives.items() if value['id']==r['id']]}
                          for r in chosen]})
        write_json(root/'status.json',{'status':'completed','successful_new_strategies':sum(r['id']!='initial' and r['status']=='completed' for r in state['records']),
            'early_stopped_new_strategies':sum(r['status']=='early_stopped' for r in state['records']),
            'quality_eligible_strategies':qualified,'selected_count':len(chosen),
            'representative_count':len(assignments),'selection_status':selection_status})
        return state
    except BaseException as exc:
        save(root,state)
        write_json(root/'status.json',{'status':'incomplete','error':f'{type(exc).__name__}: {exc}'})
        raise
    finally: runner.close()


def final_table(reference_summary, strategies):
    lines = ['| Representative(s) | Strategy | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NFE ↓ | Seconds/video ↓ |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in [{'id':'initial', 'summary':reference_summary}, *strategies]:
        s = row['summary']
        values = [('∞' if s[k] is None else f'{s[k]:.4f}') for k in
                  ('mean_psnr','mean_ssim','mean_lpips','mean_nfe','mean_seconds')]
        categories=', '.join(row.get('categories', ['reference']))
        lines.append('| ' + categories + ' | ' + row['id'] + ' | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def test_selected(config):
    root=Path(config['output_dir'])
    if (root/'search_only').exists():
        print('Search-only run: final evaluation is disabled.', flush=True)
        return
    selection=read(root/'selected_representatives.json')
    path=Path(config['test_prompts'])
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_rows(rows)
    if {r['id'] for r in rows}&{r['id'] for r in config['rows']} or {r['prompt'] for r in rows}&{r['prompt'] for r in config['rows']}:
        raise ValueError('Final test prompts must be disjoint from development prompts')
    if not selection['strategies']:
        write_json(root/'test/results.json',{'strategies':[],'representatives':{},'selection_status':selection['selection_status'],
            'note':'No valid new strategy found. No test generation or automatic budget extension.',
            'quality_report':str(root/'quality_report.json')})
        (root/'test/metrics.md').write_text('No valid new strategy found.\n', encoding='utf-8')
        return
    runner=make_evaluator(config)
    try:
        reference=Path(config.get('test_reference_results',root/'test/reference/results.json'))
        if config.get('test_reference_results'):
            ref=valid_result(reference,runner.context(rows,root/'strategies/initial',True,None))
        else:
            job={'id':'test_reference','policy':str(root/'strategies/initial'),
                 'output':str(root/'test/reference'),'initial':True}
            _,ref,error=next(evaluate_jobs(runner,[job],rows,None))
            if error is not None: raise RuntimeError(error)
        results=[]
        jobs=[]
        for r in selection['strategies']:
            jobs.append({'id':r['id'],'policy':r['policy'],'output':str(root/'test'/r['id'])})
        for job, measured, error in evaluate_jobs(runner,jobs,rows,reference):
            if error is not None: raise RuntimeError(error)
            r=next(r for r in selection['strategies'] if r['id']==job['id'])
            results.append({'id':r['id'],'categories':r['categories'],'summary':measured['summary'],
                            'quality_eligible':r['quality_eligible'],
                            'results':str(root/'test'/r['id']/'results.json')})
        order={r['id']:i for i,r in enumerate(selection['strategies'])}
        results.sort(key=lambda r:order[r['id']])
        write_json(root/'test/results.json',{'reference_results':str(reference),
            'reference_summary':ref['summary'],'strategies':results,'representatives':selection['representatives'],
            'selection_status':selection['selection_status'],'qualified_new_strategies':selection['qualified_new_strategies'],
            'selection':'Categories fixed on development data after all search rounds; unique strategies tested once. Test scores do not reselect representatives or feed the Agent.'})
        note = ('No qualifying new strategy found on the development set. Representatives were selected from all valid new strategies.\n\n'
                if selection['selection_status']=='fallback_no_eligible_new_strategy' else '')
        (root/'test/metrics.md').write_text(note+final_table(ref['summary'], results), encoding='utf-8')
    finally: runner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','test'])
    parser.add_argument('config');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=load_config(args.config)
    if args.command=='check':
        print(json.dumps({'valid':True,'videos':len(config['rows']),'round_sizes':config['round_sizes'],
                          'loss':config['loss'],'gpu_generation':False,'paid_agent_calls':False}))
    else:
        import fcntl
        root = Path(config['output_dir'])
        root.parent.mkdir(parents=True, exist_ok=True)
        with (root.parent / (root.name + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command=='run': run(config,resume=args.resume)
            else: test_selected(config)


if __name__=='__main__':main()
