"""Install/start, inspect or stop the macOS per-user demo and HTTPS tunnel services."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home()/'Library/Application Support/AspireVision'
AGENTS = Path.home()/'Library/LaunchAgents'
LABELS = ('com.aspire.vision.http', 'com.aspire.vision.tunnel')
DOMAIN = f'gui/{os.getuid()}'


def stop():
    for label in reversed(LABELS):
        subprocess.run(['launchctl','bootout',f'{DOMAIN}/{label}'], capture_output=True)
        (AGENTS/f'{label}.plist').unlink(missing_ok=True)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if subprocess.run(['launchctl','print',f'{DOMAIN}/{label}'],capture_output=True).returncode!=0:break
            time.sleep(0.2)


def status():
    config = json.loads((STATE/'service.json').read_text()) if (STATE/'service.json').exists() else {}
    pinggy = config.get('provider') == 'pinggy'
    log = STATE/('tunnel.stdout.log' if pinggy else 'tunnel.stderr.log')
    pattern = r'https://[a-z0-9.-]+\.pinggy-free\.link' if pinggy else r'https://[a-z0-9-]+\.trycloudflare\.com'
    urls = re.findall(pattern, log.read_text(errors='replace')) if log.exists() else []
    services = {}
    for label in LABELS:
        result = subprocess.run(['launchctl','print',f'{DOMAIN}/{label}'],capture_output=True,text=True)
        services[label] = {'loaded':result.returncode==0,'running':'state = running' in result.stdout}
    return {'config':config,'services':services,'last_public_url':urls[-1] if urls else None,'logs':str(STATE)}


def start(port, provider):
    cloudflared = shutil.which('cloudflared')
    if provider=='cloudflare' and not cloudflared:
        raise SystemExit('Install cloudflared first: brew install cloudflared')
    STATE.mkdir(parents=True,exist_ok=True)
    AGENTS.mkdir(parents=True,exist_ok=True)
    stop()
    if provider=='pinggy':
        key=STATE/'pinggy_key'
        if not key.exists():
            subprocess.run(['/usr/bin/ssh-keygen','-q','-t','ed25519','-N','','-C','aspire-vision-demo','-f',str(key)],check=True)
        tunnel=['/usr/bin/ssh','-T','-F','/dev/null','-p','443','-i',str(key),
                '-o','BatchMode=yes','-o','IdentityAgent=none','-o','IdentitiesOnly=yes',
                '-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={STATE/"known_hosts"}',
                '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
                '-o','ExitOnForwardFailure=yes','-R',f'0:127.0.0.1:{port}','free@free.pinggy.io']
    else:
        tunnel=[cloudflared,'--no-autoupdate','tunnel','--protocol','auto','--grace-period','5s','--url',f'http://127.0.0.1:{port}']
    commands = [
        [sys.executable,str(ROOT/'scripts/serve_vision_demo.py'),'--bind','0.0.0.0','--port',str(port)],
        tunnel,
    ]
    (STATE/'service.json').write_text(json.dumps({'repository':str(ROOT),'port':port,'provider':provider,'url_lifetime_minutes':60 if provider=='pinggy' else None},indent=2)+'\n')
    try:
        for label, command, name in zip(LABELS,commands,('http','tunnel')):
            stdout, stderr = STATE/f'{name}.stdout.log', STATE/f'{name}.stderr.log'
            stdout.write_text('');stderr.write_text('')
            spec = {'Label':label,'ProgramArguments':command,'WorkingDirectory':str(ROOT),
                    'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':10,
                    'StandardOutPath':str(stdout),'StandardErrorPath':str(stderr),
                    'EnvironmentVariables':{'PYTHONUNBUFFERED':'1','PATH':'/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin'}}
            plist = AGENTS/f'{label}.plist'
            plist.write_bytes(plistlib.dumps(spec))
            for attempt in range(6):
                boot=subprocess.run(['launchctl','bootstrap',DOMAIN,str(plist)],capture_output=True,text=True)
                if boot.returncode==0:break
                time.sleep(0.5)
            if boot.returncode:raise RuntimeError(boot.stderr.strip())
        deadline = time.monotonic()+55
        while time.monotonic()<deadline:
            report = status()
            try:
                with urlopen(f'http://127.0.0.1:{port}/',timeout=1) as response:
                    healthy = response.status == 200
            except OSError:
                healthy = False
            connected = bool(report['last_public_url']) if provider=='pinggy' else 'Registered tunnel connection' in (STATE/'tunnel.stderr.log').read_text(errors='replace')
            if healthy and report['last_public_url'] and connected:
                return report
            time.sleep(1)
        raise RuntimeError(f'Services have not become ready. Inspect logs: {STATE}')
    except Exception:
        stop()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','stop'])
    parser.add_argument('--port',type=int,default=8091)
    parser.add_argument('--provider',choices=['cloudflare','pinggy'],default='pinggy')
    args=parser.parse_args()
    if sys.platform!='darwin':parser.error('This launchd helper requires macOS; run serve_vision_demo.py directly elsewhere')
    if args.action=='start':report=start(args.port,args.provider)
    elif args.action=='stop':stop();report=status()
    else:report=status()
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
