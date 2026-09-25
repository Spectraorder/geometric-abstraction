"""Robot-only placement reach scan; an optimistic planning bound, not task success."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import mujoco

from aspire_so100.stacking_env import StackingEnv, TABLE_HEIGHT
from run_vision_trial import plan_tool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    records=[]
    with TemporaryDirectory() as directory:
        env=StackingEnv(Path(directory), scene_task='upright_thin_prism', object_count=2)
        try:
            mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
            mujoco.mj_forward(env.model, env.data)
            for x in (-.08,-.04,0.,.04,.08):
                for y in (-.20,-.24,-.28):
                    valid=[]
                    release_valid=[]
                    position_valid=[]
                    samples=[]
                    for height in np.arange(.03,.361,.01):
                        # Last payload center is half its height below the tower
                        # top. Conservative 30 mm grasp-site offset, plus 30 mm
                        # approach/withdraw clearance, for 30/60 mm blocks.
                        errors=[];tilts=[];release_errors=[];release_tilts=[]
                        for payload in (.03,.06):
                            for clearance in (0.,.03):
                                plan=plan_tool(env,[x,y,TABLE_HEIGHT+height-payload/2+.03+clearance])
                                errors.append(plan['position_error'])
                                tilts.append(float(np.arccos(np.clip(plan['tool_rotation'][2,1],-1,1))))
                                if clearance == 0.:
                                    release_errors.append(errors[-1])
                                    release_tilts.append(tilts[-1])
                        samples.append({'height_m':float(height),'max_position_error_m':max(errors),
                                        'max_tilt_deg':math.degrees(max(tilts)),
                                        'release_tilt_deg':math.degrees(max(release_tilts))})
                        if max(errors)<=.003 and max(tilts)<=math.radians(8):
                            valid.append(float(height))
                        if max(release_errors)<=.003 and max(release_tilts)<=math.radians(8):
                            release_valid.append(float(height))
                        if max(errors)<=.003:
                            position_valid.append(float(height))
                    records.append({'xy':[x,y], 'reachable_tower_height_m':max(valid,default=0.),
                                    'release_only_height_m':max(release_valid,default=0.),
                                    'position_only_height_m':max(position_valid,default=0.),
                                    'height_samples':samples})
        finally:
            env.close()
    best=max(records,key=lambda r:r['reachable_tower_height_m'])
    height=best['reachable_tower_height_m']
    result={'method':'robot-only IK grid; 10 mm height resolution; soft downward orientation',
            'position_tolerance_m':.003,'tilt_tolerance_deg':8,
            'grasp_offset_m':.03,'approach_clearance_m':.03,
            'best':{k:v for k,v in best.items() if k != 'height_samples'},'samples':records,
            'best_release_only_height_m':max(r['release_only_height_m'] for r in records),
            'best_position_only_height_m':max(r['position_only_height_m'] for r in records),
            'minimum_count_60mm_upright':math.ceil(round(height/.06,8)),
            'count_30mm_upright':math.ceil(round(height/.03,8)),
            'caveat':'Sampled feasible lower estimate, not a proof of global maximum. Counts are optimistic geometric estimates; no collisions, grasp feasibility, stability, or mixed-shape support certified.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'samples'},indent=2))


if __name__=='__main__':
    main()
