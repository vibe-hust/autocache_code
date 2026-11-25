"""Let a RunPod worker handle an OPAL Agent call while generation stays on Trento."""
import json
from pathlib import Path
import sys
import time


request_path, workspace = map(Path, sys.argv[1:3])
request = json.loads(request_path.read_text())
(workspace / 'bridge_ready.json').write_text(json.dumps({'request': str(request_path)}))
try:
    while not (workspace / 'bridge_done.json').is_file():
        time.sleep(2)
    result = json.loads((workspace / 'bridge_done.json').read_text())
    if result['exit_code']:
        raise RuntimeError('RunPod Agent failed. See agent_cli_stderr.txt.')
    session = Path(request['run_root']) / 'agent_sessions/session.json'
    session.parent.mkdir(parents=True, exist_ok=True)
    temporary = session.with_suffix('.tmp')
    temporary.write_text(json.dumps({'run_root': request['run_root'],
        'provider': {k: request['agent'].get(k) for k in ('model', 'profile', 'reasoning_effort')},
        'thread_id': result['thread_id']}))
    temporary.replace(session)
finally:
    (workspace / 'bridge_closed.json').write_text('{}')
