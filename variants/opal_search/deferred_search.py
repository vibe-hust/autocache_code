"""All proposals precede generation; deployed no-feedback ablation controller."""
from pathlib import Path
from reactive_samplers.report import write_json
from .agent import invoke, safe_child
from .deferred_input import round_input
from .diagnostics import diagnose, validate_request
from .evaluate import read, valid_result
from .parallel import make_evaluator, evaluate_jobs
from .policy import freeze, source_hash, structure_hash, Runtime, tree
from .selection import QUALITY_GATE, REPRESENTATIVE_METRICS, quality_report, select_representatives, paired
from .search import save

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
            'results':str(root/'evaluations/initial/results.json')}], 'frontier':[], 'diagnostics_used':0, 'deferred_diagnostics':[]}
        save(root,state)
    runner=None
    try:
        initial=state['records'][0]
        for number,target in enumerate(config['round_sizes'],1):
            if str(number) in state.get('round_outcomes',{}): continue
            while True:
                completed=sum(r['round']==number and r['status']=='pending' for r in state['records'])
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
                        validate_request(payload,config['rows'],config)
                        if not any(r['id']==payload['strategy_id'] for r in state['records']):
                            raise ValueError('Diagnostic must name initial or an accepted proposal')
                        request=dict(payload)
                        if request['kind']=='probe':
                            if state['diagnostics_used']>=config['diagnostic_budget']: raise ValueError('Probe budget exhausted')
                            source=safe_child(workspace,request['probe_path']).read_text()
                            tree(source,'signals')
                        archive=root/'diagnostics'/f'{folder.name}_{workspace.name}'
                        archive.mkdir(parents=True,exist_ok=True)
                        if request['kind']=='probe':
                            (archive/'probe.py').write_text(source)
                            request['probe_path']='probe.py'
                            state['diagnostics_used']+=1
                        write_json(archive/'request.json',request)
                        state.setdefault('deferred_diagnostics',[]).append(
                            {'request':request,'workspace':str(archive),'status':'queued'})
                        save(root,state)
                        write_json(workspace/'feedback.json',{'diagnostic':{
                            'status':'queued','kind':request['kind'],'strategy_id':request['strategy_id']}})
                        continue
                    items=payload['strategies']
                    if not isinstance(items,list) or len(items)!=target-completed:
                        raise ValueError('Submit exactly the requested number of new packages')
                    known={r['structure_sha256'] for r in state['records'] if r['status'] in ('pending','completed')}
                    ids={r['id'] for r in state['records']};validated=[]
                    for item in items:
                        sid=item['id']
                        if not isinstance(sid,str) or not sid or len(sid)>80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid) or sid in ids:
                            raise ValueError('Unique safe strategy ID required')
                        parent=next((r for r in state['records'] if r['id']==item['parent'] and r['status']=='pending'),None)
                        if parent is None: raise ValueError('Parent must name an accepted unmeasured strategy or initial')
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
                        state['records'].append(record);save(root,state)
                except (ValueError,KeyError,TypeError,SyntaxError,FileNotFoundError) as exc:
                    write_json(workspace/'feedback.json',{'submission_error':f'{type(exc).__name__}: {str(exc)[-2000:]}'})
                    continue
            save(root,state)
        # No generator or metric evaluator exists until every proposal round is fixed.
        runner=evaluator or make_evaluator(config)
        if initial['status']!='completed':
            write_json(root/'status.json',{'status':'running','stage':'reference'})
            if config.get('reference_results'):
                data=valid_result(config['reference_results'],runner.context(config['rows'],initial['policy'],True,None))
                initial['results']=str(Path(config['reference_results']).resolve())
            else:
                data=runner.run(config['rows'],initial['policy'],Path(initial['results']).parent,initial=True)
            if any(r['nfe']!=config['generation']['reference_steps']*(2 if config['generation']['model']=='flux3' else 1) for r in data['images']): raise ValueError('Initial reference must be all NEW')
            initial.update(status='completed',summary=data['summary']);save(root,state)
        pending=[r for r in state['records'] if r['id']!='initial' and r['status']=='pending']
        jobs=[{'id':r['id'],'policy':r['policy'],'output':str(Path(r['results']).parent)} for r in pending]
        write_json(root/'status.json',{'status':'running','stage':'deferred_evaluation',
            'strategies':[job['id'] for job in jobs], 'candidate_gpus':config.get('evaluation_gpus',[0,1])})
        for job, measured, error in evaluate_jobs(runner,jobs,config['rows'],initial['results']):
            record=next(r for r in pending if r['id']==job['id'])
            if error is None:
                record.update(status='completed',summary=measured['summary'])
            else:
                record.update(status='failed',error=error)
            save(root,state)
        for item in state.get('deferred_diagnostics',[]):
            if item['status']!='queued': continue
            output=Path(item['workspace'])
            write_json(root/'status.json',{'status':'running','stage':'deferred_diagnostics',
                'strategy':item['request']['strategy_id'],'kind':item['request']['kind']})
            try:
                diagnose(item['request'],output,output,state['records'],config['rows'],
                    runner,initial['results'],config)
                item.update(status='completed',result=str(output/'result.json'))
            except (ValueError,KeyError,TypeError,SyntaxError,FileNotFoundError) as exc:
                item.update(status='failed',error=f'{type(exc).__name__}: {str(exc)[-2000:]}')
                write_json(output/'error.json',{'error':item['error']})
            save(root,state)
        # Parent results may arrive in any order; comparisons are computed after all evaluations.
        for record in state['records']:
            if record['id']=='initial' or record['status']!='completed': continue
            data=valid_result(record['results'])
            record['feedback']={'id':record['id'],
                'versus_initial':paired(data['images'],valid_result(initial['results'])['images'])}
            parent=next(r for r in state['records'] if r['id']==record['parent'])
            if parent['id']!='initial' and parent['status']=='completed':
                record['feedback']['versus_parent']=paired(data['images'],valid_result(parent['results'])['images'])
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
    finally:
        if runner is not None: runner.close()
