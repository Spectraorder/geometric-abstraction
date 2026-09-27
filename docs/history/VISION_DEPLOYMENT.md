> Historical record. Commands and status statements describe that experiment;
> see the [current documentation](../README.md) before running anything.

# Visualization: static public mirror, Mac LAN server and background tunnel

The recorded deployment published a [static Vercel demo](https://aspire-so100-vision.vercel.app/) that does not depend on the Mac remaining online. The Mac also hosted a LAN server and an SSH/443 tunnel. All served `web/vision/`: seven actual physics recordings, two rounds of candidates/prompts, results and hashes.

[deployment-status.json](../deployment-status.json) records the deployment checks: HTTP 200 for the homepage, 206 video range requests, matching complete MP4 hashes, and browser playback reaching 18.24 seconds of a 37.6-second clip. These are historical checks, not a current uptime claim.

## Local and LAN access

Serving the page requires Python 3.10+ and no MuJoCo installation:

```bash
python3 scripts/serve_vision_demo.py --bind 0.0.0.0 --port 8091
```

Open `http://127.0.0.1:8091/` locally or `http://<Mac-LAN-IP>:8091/` on the LAN. On the Mac, `ipconfig getifaddr en0` reports the Wi-Fi address. LAN addresses are not directly reachable from the public Internet.

The server exposes selected static files, not a simulation-control API. HTTP/1.1 Range requests support seeking. Path traversal, escaping symlinks, hidden files/directories and directory listings are rejected. Raw experiment directories, model sessions and credentials are not published.

## Mac background service and public tunnel

The recorded setup installed two per-user LaunchAgents: `com.aspire.vision.http` and `com.aspire.vision.tunnel`. They survive terminal closure, restart after process exit, and start at user login.

```bash
python3 scripts/vision_demo_service.py start --provider pinggy --port 8091
python3 scripts/vision_demo_service.py status
python3 scripts/vision_demo_service.py stop
```

The historical tunnel URL was `https://fgsjk-185-220-238-74.run.pinggy-free.link/`. At the time of deployment, free Pinggy connections expired after 60 minutes; reconnecting or restarting produced a new address. Read the service's latest `last_public_url`; use the Vercel mirror for a persistent shared link.

The Mac endpoint requires the machine to remain awake, online and logged in. Sleep or loss of connectivity affects that endpoint, but not the static mirror. SSH forwards only the display port, without remote shell or agent forwarding. A dedicated local key and separate known_hosts remain on the Mac and are not committed.

Logs/state: `~/Library/Application Support/AspireVision/`. Launch configuration: `~/Library/LaunchAgents/`. `stop` removes this service's launch configuration and stops its processes, retaining recordings. Stop before moving the workspace, then start from the new path.

Cloudflare is also supported. Repeated connection timeouts prevented its use for the recorded delivery:

```bash
brew install cloudflared
python3 scripts/vision_demo_service.py start --provider cloudflare --port 8091
```

Quick Tunnels use temporary addresses without an SLA. Neither a domain nor a paid tunnel plan was purchased. A running process and a logged URL do not establish reachability; verify the actual endpoint.

Provider references: [Pinggy help](https://pinggy.io/help/) and [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).

## Updating the static mirror

Project: `lucas-projects-a9e8b0a3/aspire-so100-vision`. It hosts static display files, not online robot control.

```bash
cd web/vision
vercel link --yes --project aspire-so100-vision --scope lucas-projects-a9e8b0a3
vercel deploy --prod --yes --scope lucas-projects-a9e8b0a3
```

Deployment requires the corresponding Vercel login. `.vercel/` and `.env*` are ignored. The recorded publication used the CLI; automatic GitHub linkage did not succeed, so a Git push alone did not redeploy that mirror.

## Exporting the display package again

The source campaign must contain both rounds, seeds 81–83, reserved seeds 91–92, and the blackout control. The exporter copies only allowlisted assets and requires a new output directory:

```bash
python3 scripts/export_vision_demo.py \
  --campaign outputs/vision_from_scratch_01 \
  --output outputs/vision_public_export
python3 scripts/serve_vision_demo.py --root outputs/vision_public_export --port 8092
```

Each clip records policy/video hashes, camera calls and policy time; playback is explicitly marked at 2× speed. See the [visual experiment guide](VISION_FROM_SCRATCH.md) for rerunning a frozen policy or learning from an empty task-skill library.
