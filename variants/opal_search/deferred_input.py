"""Proposal history and code errors only; no result-dependent feedback."""
from pathlib import Path
from reactive_samplers.report import write_json
from .policy import FILES
from .selection import QUALITY_GATE, target_description

def round_input(root, workspace, round_number, needed, state, config, feedback):
    root, workspace = Path(root), Path(workspace)
    task_file = 'deferred_first_round.md' if round_number == 1 else 'deferred_later_rounds.md'
    def proposal(record):
        return {**{key:record[key] for key in ('id','round','parent','change','evidence','hypothesis') if key in record},
                'code':{name:(Path(record['policy'])/name).read_text() for name in FILES}}
    value = {'round':round_number,
        'task':(Path(__file__).parent/'prompts'/task_file).read_text().replace('{{TARGETS}}',target_description()),
        'requested_new_strategies':needed,
        'remaining_strategy_slots':sum(config['round_sizes'][round_number:])+needed,
        'max_replacement_rounds':config['max_replacement_rounds'],
        'initial_strategy':proposal(state['records'][0]),
        'previous_proposals':[proposal(r) for r in state['records'] if r['id']!='initial'],
        'new_feedback':[{key:item[key] for key in ('error','submission_error','diagnostic') if key in item}
                        for item in feedback if any(key in item for key in ('error','submission_error','diagnostic'))],
        'remaining_diagnostic_budget':config['diagnostic_budget']-state['diagnostics_used'],
        'diagnostic_sample_ids':[row['id'] for row in config['rows']],
        'queued_diagnostics':[{key:item['request'][key] for key in ('kind','strategy_id','sample_ids','steps')}
                              for item in state.get('deferred_diagnostics',[])],
        'quality_requirement':QUALITY_GATE, 'loss':config['loss'],
        'evaluation_timing':'All nine proposal rounds are fixed before any initial or candidate generation/evaluation. No measured feedback is available during proposals.',
        'on_demand':{'project_code':{'path':str(Path(__file__).resolve().parents[2]),
            'description':'Read-only sampling, cache and policy-interface code. Do not read results from other experiments.'}}}
    write_json(workspace/'round_input.json',value)
    return value
