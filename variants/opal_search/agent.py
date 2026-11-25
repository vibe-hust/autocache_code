"""Round evidence, a persistent Agent session, and bounded machine-readable submissions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from reactive_samplers.report import write_json
from .evaluate import read
from .policy import FILES, interface_description
from .selection import QUALITY_GATE, quality_report, target_description


def safe_child(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute(): raise ValueError('Expected workspace-relative path')
    result = (root/relative).resolve()
    if not result.is_relative_to(root): raise ValueError('Path escapes workspace')
    return result


def strategy_entry(record, frontier):
    return {'id': record['id'],
            'code': {name: (Path(record['policy']) / name).read_text() for name in FILES},
            'scores': record['summary'], 'quality_gate': quality_report(record['summary']),
            'is_pareto': record['id'] in frontier}


def round_input(root, workspace, round_number, needed, state, config, feedback):
    root, workspace = Path(root), Path(workspace)
    task_file = 'first_round.md' if round_number == 1 else 'later_rounds.md'
    if config.get('fixed_signals'): task_file = 'fixed_'+task_file
    front, qualified_extra = [], []
    for record in state['records']:
        if record['status'] != 'completed': continue
        is_front = record['id'] in state['frontier']
        gate = quality_report(record['summary'])
        if not is_front and (record['id']=='initial' or not gate['eligible']): continue
        entry = strategy_entry(record, state['frontier'])
        (front if is_front else qualified_extra).append(entry)
    routes = {}
    for record in state['records']:
        if record['status'] != 'completed': continue
        measured = read(record['results'])
        routes[record['id']] = {
            'code': {name: str(Path(record['policy']) / name) for name in FILES},
            'per_image_metrics': record['results'],
            'samples': {image['id']: {
                'image': image['image'],
                'trajectory_and_observations': image['trajectory'],
                'observations_available': image.get('observations_available', not measured.get('provenance'))}
                for image in measured['images']}}
    feedback = [item for item in feedback
                if any(key in item for key in ('error', 'submission_error', 'diagnostic'))]
    value = {'round': round_number,
        'task': (Path(__file__).parent/'prompts'/task_file).read_text().replace('{{TARGETS}}', target_description()).replace('Qwen-Image-2.1', {'qwen21':'Qwen-Image-2.1', 'flux3':'FLUX3-Image-Klein-Base', 'flux':'FLUX.1-dev'}[config['generation']['model']]),
        'requested_new_strategies': needed,
        'remaining_strategy_slots': sum(config['round_sizes'][round_number:])+needed,
        'max_replacement_rounds': config['max_replacement_rounds'],
        'remaining_diagnostic_budget': config['diagnostic_budget']-state['diagnostics_used'],
        'pareto': front, 'qualified_outside_pareto':qualified_extra,
        'new_feedback': feedback, 'quality_requirement':QUALITY_GATE,
        'qualified_new_strategy_count':sum(r['status']=='completed' and r['id']!='initial'
            and quality_report(r['summary'])['eligible'] for r in state['records']),
        'on_demand': {
            'history_strategies': {
                'path': str(root/'leaderboard.json'),
                'description': 'Completed historical strategies outside the default pool, using the same id/code/scores/quality_gate/is_pareto format as pareto.'},
            'strategy_materials': {
                'description': 'Direct paths for every completed strategy. per_image_metrics contains per-image five quality scores, NFE, time and loss plus strategy means. Each sample points directly to its image and JSONL trajectory with observations, added signals, actions, conditions and operands. Imported initial references have actions only when observations_available is false.',
                'strategies': routes},
            'project_code': {
                'path': str(Path(__file__).resolve().parents[2]),
                'description': 'Read-only sampling, cache and policy-interface implementation.'}}}
    if config.get('reference_notes'):
        value['reference_notes'] = config['reference_notes']
    value['quality_rule_update'] = target_description()
    write_json(workspace/'round_input.json', value)
    return value


def invoke(root, workspace, config):
    root, workspace = Path(root), Path(workspace)
    session = root/'agent_sessions/session.json'
    first = not session.exists()
    request = {'run_root': str(root.resolve()), 'workspace': str(workspace.resolve()),
               'input': str((workspace/'round_input.json').resolve()), 'first_call': first,
               'agent': config['agent']}
    if first:
        prompt_name = 'deferred_search.md' if config.get('feedback_mode')=='deferred' else 'fixed_search.md' if config.get('fixed_signals') else 'search.md'
        request['instructions'] = (Path(__file__).parent/'prompts'/prompt_name).read_text().replace('{{TARGETS}}', target_description()).replace('Qwen-Image-2.1', {'qwen21':'Qwen-Image-2.1', 'flux3':'FLUX3-Image-Klein-Base', 'flux':'FLUX.1-dev'}[config['generation']['model']])
        request['interface'] = interface_description(config['generation']['model'], config.get('observation_fields'))
        if config.get('fixed_signals'):
            request['interface'].update(editable_files=['updater.py'], fixed_signals=(Path(__file__).parent/'templates/signals.py').read_text())
        request['generation'] = config['generation']
        request['state_tensor_bytes_per_image'] = config['state_tensor_bytes']
        if config.get('feedback_mode') == 'deferred':
            request['instructions'] += ('\nAll proposal rounds precede any generation. Only code/interface errors and '
                                       'previous proposals are available. Do not inspect other runs. Diagnostic '
                                       'requests are queued and execute only after every proposal and candidate '
                                       'evaluation is complete; no diagnostic results are returned during proposals. '
                                       'Submit the requested strategies. No measured initial '
                                       'reference or candidate metrics are available during proposals.')
    write_json(workspace/'request.json', request)
    command = config['agent'].get('command') or [sys.executable, str(Path(__file__).with_name('agent_cli.py')), '{request}', '{workspace}']
    command = [part.replace('{request}', str((workspace/'request.json').resolve())).replace('{workspace}', str(workspace.resolve())).replace('{project_root}',str(Path(__file__).resolve().parents[2])).replace('{python}',sys.executable) for part in command]
    before = time.perf_counter()
    usage = {'input_tokens': 0, 'output_tokens': 0, 'calls_with_usage': 0, 'calls_without_usage': 1}
    try:
        process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=config['agent'].get('timeout_seconds',1500))
        except BaseException:
            # Stop this invocation and its CLI child, not another experiment's Agent.
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.communicate()
            raise
        (workspace/'stdout.txt').write_text(stdout)
        (workspace/'stderr.txt').write_text(stderr)
        if process.returncode: raise RuntimeError('Agent process failed: '+stderr[-2000:])
    finally:
        events = workspace/'agent_events.jsonl'
        if events.exists():
            for line in events.read_text().splitlines():
                try: event = json.loads(line)
                except ValueError: continue
                u = event.get('usage', {}) if event.get('type') == 'turn.completed' else {}
                if all(type(u.get(k)) is int for k in ('input_tokens','output_tokens')):
                    for key in ('input_tokens', 'output_tokens'): usage[key] += u[key]
                    usage.update(calls_with_usage=1, calls_without_usage=0)
        write_json(workspace/'usage.json', {**usage, 'seconds': time.perf_counter()-before})
    choices = [n for n in ('submission.json','diagnostic_request.json') if (workspace/n).is_file()]
    if len(choices) != 1: raise ValueError('Submit exactly one of submission.json or diagnostic_request.json')
    return choices[0], read(workspace/choices[0])
