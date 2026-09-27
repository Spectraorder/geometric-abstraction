"""Compare old/new thread deflection under a fixed engineering load fixture."""
from pathlib import Path
from unittest.mock import patch
import json
import mujoco
import numpy as np
from aspire_so100.fasten_screws import FastenScrewsEnv
from aspire_so100.fasteners import SCENE_VERSION, ROBODOJO_LEGACY_VERSION
import argparse
from tempfile import TemporaryDirectory
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
temporary=TemporaryDirectory(prefix='aspire-thread-audit-')
out={}
for version in [ROBODOJO_LEGACY_VERSION,SCENE_VERSION]:
 e=FastenScrewsEnv(Path(temporary.name)/version,scene_version=version);e.reset(5101);m,d=e.model,e.data;n=e.nuts['red']
 e._set_freejoint_pose('red_nut',d.xpos[e.bolts['red']]+[0,0,e.entry_z],0);mujoco.mj_forward(m,d)
 with patch.object(e,'_contacts',return_value=e.jaws):e._thread_forces()
 e._step_for(1)
 errors=[]
 for i in range(500):
  e._thread_forces();d.xfrc_applied[n,:3]+=[3,0,-6];d.xfrc_applied[n,4]+=.025;mujoco.mj_step(m,d)
  delta,rot,_=e._pose('red','red');thread=e.threads.get('red')
  if thread:errors.append([abs(delta[2]-(thread['entry_z']+e.thread_lead*thread['twist'])),np.linalg.norm(delta[:2]),np.arccos(np.clip(rot[2,2],-1,1))])
 before=e.pair_status('red');z_before=d.xpos[n,2];e._step_for(2);z_after=d.xpos[n,2]
 out[version]=dict(max_error=np.max(errors,axis=0).tolist(),rebound_m=float(z_after-z_before),before=before,after=e.pair_status('red'),warnings=d.warning.number.tolist())
 print(version,out[version],flush=True);e.close()
print(json.dumps(out,indent=2),flush=True)

(a.output/'load_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
temporary.cleanup()
