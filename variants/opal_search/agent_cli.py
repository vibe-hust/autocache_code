"""Dedicated OPAL Codex/DeepSeek launcher; session never shared with A training."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main(request_path, workspace):
    request = json.loads(Path(request_path).read_text()); cfg = request['agent']
    root, workspace = Path(request['run_root']), Path(workspace).resolve()
    input_path = Path(request['input']); data = json.loads(input_path.read_text())
    if 'budget_rule' in data:
        del data['budget_rule']
        input_path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    session_file = root/'agent_sessions/session.json'
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session = json.loads(session_file.read_text()) if session_file.exists() else {}
    identity = {k:cfg.get(k) for k in ('model','profile','reasoning_effort')}
    if session and (session['run_root'] != str(root) or session['provider'] != identity):
        raise ValueError('Agent session belongs to another experiment/provider')
    env = os.environ.copy()
    if cfg.get('proxy_url'):
        for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):
            env[key] = cfg['proxy_url']
        for key in ('ALL_PROXY','all_proxy'):
            env.pop(key, None)
        env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1'
    env['PATH'] = str(Path.home()/'.local/bin')+':'+env.get('PATH','')
    profile = cfg.get('profile', 'codex-personal')
    if profile == 'deepseek-api':
        env['CODEX_HOME'] = str(Path.home()/'.codex-reactive-samplers-deepseek')
        if not env.get('DEEPSEEK_API_KEY'):
            env['DEEPSEEK_API_KEY'] = (Path.home()/'.config/reactive-samplers-agent/deepseek-api-key').read_text().strip()
    elif profile == 'codex-personal':
        env['CODEX_HOME'] = cfg.get('codex_home', str(Path.home()/'.codex-reactive-samplers-personal'))
        env.pop('OPENAI_API_KEY',None); env.pop('DEEPSEEK_API_KEY',None)
    else:
        raise ValueError('Unsupported profile')
    if not cfg.get('model'): raise ValueError('Explicit Agent model required')
    if session:
        prompt = f'Continue OPAL search. Read {request["input"]}. Follow the original task and interface. Write submission.json with the requested number of new strategy packages, or diagnostic_request.json. Current workspace: {workspace}.'
    else:
        prompt = f'Read {Path(request_path).resolve()} for the full task and interface, then its input file. Work in {workspace}. Submit strategy packages or a diagnostic request as specified.'
    command = [cfg.get('executable','codex'),'exec','-m',cfg['model'],
        '-c','model_reasoning_effort='+json.dumps(cfg.get('reasoning_effort','high')),
        '--json','--sandbox','danger-full-access','--skip-git-repo-check','-C',str(workspace)]
    if session: command += ['resume',session['thread_id']]
    command.append(prompt)
    events = workspace/'agent_events.jsonl'; errors = workspace/'agent_cli_stderr.txt'
    with events.open('w') as out, errors.open('w') as err:
        result = subprocess.run(command, cwd=workspace, env=env, stdout=out, stderr=err,
                                timeout=cfg.get('timeout_seconds',1500)-5)
    thread = session.get('thread_id')
    for line in events.read_text().splitlines():
        try: event = json.loads(line)
        except ValueError: continue
        thread = event.get('thread_id',thread)
    if thread:
        temporary = session_file.with_suffix('.tmp')
        temporary.write_text(json.dumps({'run_root':str(root),'provider':identity,'thread_id':thread}))
        temporary.replace(session_file)
    if result.returncode or not thread: raise RuntimeError('Agent failed; see this call\'s stderr log')


if __name__ == '__main__': main(*sys.argv[1:3])
