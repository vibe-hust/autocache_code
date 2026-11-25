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
from .selection import QUALITY_GATE, quality_report


def safe_child(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute(): raise ValueError('Expected workspace-relative path')
    result = (root/relative).resolve()
    if not result.is_relative_to(root): raise ValueError('Path escapes workspace')
    return result


def strategy_entry(record, frontier):
    stopped = record['status'] == 'early_stopped'
    entry = {'id': record['id'],
             'code': {name: (Path(record['policy']) / name).read_text() for name in FILES},
             'scores': record['screening_summary'] if stopped else record['summary'],
             'quality_gate': None if stopped else quality_report(record['summary']),
             'is_pareto': record['id'] in frontier}
    if stopped:
        entry.update(status='early_stopped', evaluation_scope='Fixed first 20 development samples only')
    return entry


def round_input(root, workspace, round_number, needed, state, config, feedback):
    root, workspace = Path(root), Path(workspace)
    task_file = 'first_round.md' if round_number == 1 else 'later_rounds.md'
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
        if record['status'] not in ('completed', 'early_stopped'):
            continue
        measured = read(record['results'])
        routes[record['id']] = {
            'code': {name: str(Path(record['policy']) / name) for name in FILES},
            'per_video_metrics': record['results'],
            'samples': {video['id']: {
                'video': video['video'],
                'trajectory_and_observations': video['trajectory'],
                'observations_available': video.get('observations_available', False)}
                for video in measured['videos']}}
    feedback = [item for item in feedback
                if item.get('status') == 'early_stopped'
                or any(key in item for key in ('error', 'submission_error', 'diagnostic'))]
    value = {'round': round_number,
        'task': (Path(__file__).parent/'prompts'/task_file).read_text(),
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
                'description': 'Historical strategies outside this request default strategy pool. Early-stopped entries are marked and contain first-20 means; complete entries use full-set means.'},
            'strategy_materials': {
                'description': 'Direct file paths for every completed or early-stopped strategy. per_video_metrics contains per-video PSNR, SSIM, LPIPS, NFE and time plus mean scores. For early-stopped strategies, screening.json contains only the fixed first 20 samples and labels that scope. Each samples entry points directly to its video and JSONL node records. For step-policy records, each JSONL line contains the shared observation, requested action and reason, plus branches[0] (conditional) and branches[1] (unconditional) actual executions. Older branch-policy records store observation and reason inside each branch. The initial imported control has actions only when observations_available is false. No intermediary file must be opened to locate these records.',
                'strategies': routes},
            'project_code': {
                'path': str(Path(__file__).resolve().parents[2]),
                'description': 'Read-only sampling, cache and policy-interface implementation.'}}}
    if config.get('reference_notes'):
        value['reference_notes'] = config['reference_notes']
    value['budget_rule'] = ('This replaces the earlier successful-strategy quota: failures consume submission slots. '
        'Each round permits its initial submission plus at most three replacement submissions for failed slots; '
        'then advance using successful archived strategies, even below quota. Closed rounds must not be backfilled. '
        'You do not need to build your own benchmark to trial strategies before submission; '
        'the framework handles generation and evaluation after submission.')
    value['quality_rule_update'] = ('The current baseline development-mean quality scores are PSNR: 25.1022, '
        'SSIM: 0.8520, LPIPS: 0.0904. Reduce full backbone evaluations and generation time while '
        'strictly improving all three quality scores. Final qualification requires all four '
        'development-mean conditions: NFE <= 44 AND PSNR > 25.1022 AND SSIM > 0.8520 AND LPIPS < 0.0904. '
        'Use the framework\'s measured scores and gaps from '
        'each round to guide further revisions toward these goals. '
        'Nonqualifying evaluated strategies remain '
        'valid search evidence and stay in the default input when Pareto-optimal; their gaps guide revisions. '
        'All qualifying new strategies are also default evidence: pareto plus qualified_outside_pareto form a '
        'deduplicated union, so qualified strategies outside the frontier are never omitted. '
        'Per-round Pareto selection is unchanged. Only after all search rounds, select final representatives '
        'among all qualifying new strategies: lowest LPIPS, lowest generation time, and lowest existing loss. '
        'Categories may share a strategy; each unique strategy is tested once. If no new strategy qualifies, select the same three categories from all completed new strategies and explicitly report no qualifying strategy. '
        'Quality gaps are distances to strict boundaries; equality does not pass those three tests. The NFE boundary is inclusive: 44 passes.')
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
        request['instructions'] = (Path(__file__).parent/'prompts/search.md').read_text()
        request['interface'] = interface_description()
        request['generation'] = config['generation']
        request['state_tensor_bytes_per_video'] = config['state_tensor_bytes']
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
        usage['transport_returncode'] = process.returncode
        # A final stream failure may follow a completed on-disk handoff.
        # The existing JSON, package and policy checks still decide its validity.
        handoff = any((workspace / name).is_file() for name in ('submission.json', 'diagnostic_request.json'))
        if process.returncode and not handoff:
            raise RuntimeError('Agent process failed: '+stderr[-2000:])
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
