"""Run Codex on RunPod using its existing login and relay only experiment files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--run-root', action='append', required=True)
parser.add_argument('--local-root', required=True, type=Path)
parser.add_argument('--codex', required=True)
parser.add_argument('--codex-home', required=True)
parser.add_argument('--ssh-host', default='trento')
args = parser.parse_args()


def remote(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        args.ssh_host, 'python3 -'], input=code, text=True, capture_output=True, timeout=60)
    result.check_returncode()
    return result.stdout


def handle(root, relative):
    workspace = root + '/' + relative
    local_run = args.local_root / (Path(root).parent.name + '-' + hashlib.sha256(root.encode()).hexdigest()[:12])
    local = local_run / relative
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', args.ssh_host + ':' + workspace + '/', str(local) + '/'], check=True)
    request = json.loads((local / 'request.json').read_text())
    session_path = local_run / 'agent_session.json'
    session = json.loads(session_path.read_text()) if session_path.exists() else {}
    cfg = request['agent']
    identity = {k: cfg.get(k) for k in ('model', 'profile', 'reasoning_effort')}
    if session and session['identity'] != identity:
        raise ValueError('Cannot mix Agent identities in one run')
    request['input'] = str(local / 'round_input.json')
    request['workspace'] = str(local)
    request['run_root'] = str(local_run)
    (local / 'request.json').write_text(json.dumps(request, indent=2))
    prompt = (
        f'Continue this OPAL experiment. Read {local / "request.json"} and its input file. '
        f'Write submission.json or diagnostic_request.json in {local}. '
        f'You run on the RunPod relay. Paths in evidence and generation settings refer to {args.ssh_host}. '
        f'Read on-demand evidence and read-only project code with ssh {args.ssh_host}. '
        'Write all new policy packages or probe files locally under this workspace. '
        'The bridge transfers them for framework evaluation on the remote cluster. '
        'Use the original task and interface. Authentication remains managed by the existing Codex login.'
    )
    command = [args.codex, 'exec', '-m', cfg['model'], '-c',
        'model_reasoning_effort=' + json.dumps(cfg['reasoning_effort']), '--json',
        '--sandbox', 'danger-full-access', '--skip-git-repo-check', '-C', str(local)]
    if session:
        command += ['resume', session['thread_id']]
    command.append(prompt)
    env = {**os.environ, 'CODEX_HOME': args.codex_home}
    env.pop('OPENAI_API_KEY', None)
    env.pop('DEEPSEEK_API_KEY', None)
    events = local / 'agent_events.jsonl'
    with events.open('w') as output, (local / 'agent_cli_stderr.txt').open('w') as errors:
        process = subprocess.Popen(command, cwd=local, env=env, stdout=output,
            stderr=errors, start_new_session=True)
        try:
            exit_code = process.wait(timeout=cfg.get('timeout_seconds', 1500) - 30)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    thread_id = session.get('thread_id')
    for line in events.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        thread_id = event.get('thread_id', thread_id)
    if thread_id:
        session_path.write_text(json.dumps({'identity': identity, 'thread_id': thread_id}))
    choices = [name for name in ('submission.json', 'diagnostic_request.json') if (local / name).is_file()]
    if len(choices) != 1:
        exit_code = 1
    subprocess.run(['rsync', '-a', '--exclude=request.json', '--exclude=round_input.json',
        '--exclude=bridge_ready.json', '--exclude=bridge_done.json', '--exclude=bridge_closed.json',
        str(local) + '/', args.ssh_host + ':' + workspace + '/'], check=True)
    done = json.dumps({'exit_code': exit_code if thread_id else 1, 'thread_id': thread_id})
    remote('from pathlib import Path\np=Path(' + repr(workspace) + ')/"bridge_done.json"\n'
           't=p.with_suffix(".tmp");t.write_text(' + repr(done) + ');t.replace(p)\n')
    print('AGENT_FINISHED', workspace, exit_code, flush=True)


while True:
    discovery = ('from pathlib import Path\nimport json\nroots=' + repr(args.run_root) + '\n'
        'pending=[]\nfor root in roots:\n p=Path(root)\n'
        ' for f in p.glob("rounds/round_*/call_*/bridge_ready.json"):\n'
        '  if not (f.parent/"bridge_done.json").exists() and not (f.parent/"bridge_closed.json").exists():\n'
        '   pending.append([root,str(f.parent.relative_to(p))])\n'
        'jobs=[Path(r).parent for r in roots if Path(r).name=="search"]\n'
        'print(json.dumps({"pending":pending,"finished":bool(jobs) and all((p/"job.exit").exists() for p in jobs)}))\n')
    try:
        state = json.loads(remote(discovery))
        if state['finished']:
            print('VALIDATION_JOBS_FINISHED', flush=True)
            break
        for root, relative in state['pending']:
            handle(root, relative)
    except Exception as exc:
        print('BRIDGE_ERROR', type(exc).__name__, str(exc)[:500], flush=True)
    time.sleep(30)
