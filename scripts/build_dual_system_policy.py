"""Add a model-controlled initial-observation gate to the frozen RGB-D skill."""
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GATE='''

def run(api):
    # These estimates are derived exclusively from the current RGB-D image.
    for version in range(3):
        frame=api.camera()
        points,valid=rgbd_points(frame)
        rgb=np.asarray(frame['rgb'],float);r,g,b=rgb.transpose(2,0,1)
        table=valid&(r>95)&(g>65)&(r>1.15*g)&(r<1.65*g)&(g>1.35*b)
        sample=points[table];visible=[];table_valid=False
        if len(sample)>=100:
            design=np.column_stack((sample[:,:2],np.ones(len(sample))))
            keep=abs(sample[:,2]-np.median(sample[:,2]))<.025
            for fitting in range(3):
                if np.count_nonzero(keep)<50:break
                plane=np.linalg.lstsq(design[keep],sample[keep,2],rcond=None)[0]
                keep=abs(sample[:,2]-design @ plane)<.003
                table_valid=True
            if table_valid:
                height=points[:,:,2]-points[:,:,:2] @ plane[:2]-plane[2]
                for color in ('blue','red','green'):
                    if np.count_nonzero(color_mask(rgb,color)&valid&(height>.004)&(height<.15))>=8:visible.append(color)
        state={'version':version,'time_s':float(frame['time']),'phase':'initial_observation',
               'visible_colors':visible,'table_fit_valid':table_valid,'grasp_confirmed':None,
               'reobservations_left':2-version,'grasp_retries_left':1}
        answer=api.call('decision',state=state)
        if answer['state_version']!=version:raise RuntimeError('Stale decision response')
        api.note('MODEL_GATE '+str(answer['action'])+' plan='+str(answer['plan_version']))
        if answer['action']=='continue':
            if not table_valid or len(visible)!=3:raise RuntimeError('Local perception guard rejected continue')
            return run_frozen_skill(api)
        if answer['action']=='reobserve' and version<2:
            api.wait(.2)
        else:raise RuntimeError('Model gate stopped before motion')
    raise RuntimeError('Observation budget exhausted')
'''

def source():
    base=(ROOT/'policies/stack_three_blocks_visual_geometry.py').read_text()
    if base.count('def run(api):')!=1:raise ValueError('Frozen policy structure changed')
    return base.replace('def run(api):','def run_frozen_skill(api):')+GATE
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(source())
