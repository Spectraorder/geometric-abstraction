"""Recompute the bundled development summaries from actual sensor fixtures."""
import hashlib,json
from pathlib import Path
import numpy as np
from aspire_so100.instance_geometry import inventory_points,inventory_table,inventory_color_mask
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).with_name('cases.json')
d=json.loads(P.read_text())
for c in d['cases']:
 p=ROOT/c['source']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=c['sensor_sha256']:raise ValueError('Sensor fixture hash changed')
 with np.load(p) as z:f={k:z[k] for k in z.files}
 colors=[];table=True
 try:
  points,valid=inventory_points(f);plane=inventory_table(f,points,valid);h=points[:,:,2]-points[:,:,:2]@plane[:2]-plane[2]
  colors=[color for color in ('blue','red','green') if np.count_nonzero(inventory_color_mask(f['rgb'],color)&valid&(h>.004)&(h<.15))>=8]
 except ValueError:table=False
 c['state'].update(visible_colors=colors,table_fit_valid=table,time_s=float(f['time']))
 c['reference_action']='continue' if table and len(colors)==3 else 'reobserve' if c['state']['reobservations_left'] else 'escalate'
P.write_text(json.dumps(d,indent=2)+'\n');print('Verified and rebuilt',len(d['cases']),'sensor-only development summaries')
