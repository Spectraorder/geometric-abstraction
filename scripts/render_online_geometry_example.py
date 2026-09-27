"""Render a recorded sensor-to-symbolic checkpoint without private simulator state."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_frame(path):
    with np.load(path) as data:
        return {key:data[key].copy() for key in data.files}


def render(replica,trial,output,checkpoint=-1):
    library=replica/'geometry.py'
    manifest=json.loads((replica/'manifest.json').read_text())
    library_hash=hashlib.sha256(library.read_bytes()).hexdigest()
    if library_hash!=manifest['geometry_sha256']:
        raise ValueError('Frozen geometry library hash mismatch')
    namespace={}
    exec(compile(library.read_text(),str(library),'exec'),namespace)
    files=sorted(trial.glob('camera_*.npz'))
    calls=[row for row in json.loads((trial/'api_calls.json').read_text()) if row['method']=='camera']
    if len(files)!=len(calls):raise ValueError('Camera file/call count differs')
    record=json.loads((trial/'geometry_records.json').read_text())[checkpoint]
    scene=namespace['GEOMETRY'].scene(load_frame(files[0]),['blue','red','green'])
    frames={};used=[]
    for path,call in zip(files,calls):
        frame=load_frame(path)
        if abs(float(frame['time'])-record['time'])<1e-6:
            view=call['params']['view'];scene.refresh(frame,view)
            frames[view]=frame;used.append(path.name)
    if 'side' not in frames:raise ValueError('Checkpoint lacks a side observation')
    for name,value in record['objects'].items():
        if not np.allclose(np.round(scene.get(name)['center']*1000,2),value['center_mm'],atol=.011):
            raise ValueError('Sensor replay differs from recorded runtime state: '+name)
    detailed=scene.packet(details=True)
    evidence={'schema':'sensor_symbolic_illustration_v1','trial':str(trial),'checkpoint':checkpoint,
        'time':record['time'],'library_sha256':library_hash,'sensor_files':used,
        'runtime_record':record,'replayed_geometry':detailed,
        'note':'Recorded runtime checkpoint, with inferred corners replayed from its frozen RGB-D library. Desired deltas come from the recorded planner. No private result or physics trajectory is read.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14.2,7.4),layout='constrained')
    grid=fig.add_gridspec(2,3,width_ratios=[1,1.28,1.22])
    rgb=fig.add_subplot(grid[0,0]);depth=fig.add_subplot(grid[1,0])
    axes=fig.add_subplot(grid[:,1],projection='3d');text=fig.add_subplot(grid[:,2]);text.axis('off')
    side=frames['side'];rgb.imshow(side['rgb']);rgb.axis('off');rgb.set_title('1  Actual side RGB',loc='left',fontweight='bold')
    z=np.where(side['valid_depth'],side['depth_m'],np.nan)
    finite=z[np.isfinite(z)]
    depth.imshow(z,cmap='viridis',vmin=np.percentile(finite,5),vmax=np.percentile(finite,95))
    depth.axis('off');depth.set_title('Calibrated depth\nSide + wrist used for this state',loc='left',fontsize=11)
    colors={'blue':'#2474bf','red':'#d54545','green':'#2e9d61'}
    all_points=[]
    for name in ['blue','red','green']:
        state=scene.get(name);vertices=state['vertices']*1000;center=state['center']*1000
        all_points.extend(vertices)
        rank=record['objects'][name]['observed_translation_rank'];style='-' if rank==3 else '--'
        side_mm=state['height']*1000
        for i in range(8):
            for j in range(i+1,8):
                if abs(np.linalg.norm(vertices[i]-vertices[j])-side_mm)<1e-5:
                    axes.plot(*np.stack([vertices[i],vertices[j]]).T,color=colors[name],ls=style,lw=1.8)
        axes.scatter(*vertices.T,color=colors[name],s=15)
        axes.scatter(*center,color=colors[name],s=40,marker='x')
        axes.text(*(center+np.array([22,0,0])),name+'  rank '+str(rank)+'/3',color=colors[name],fontsize=9)
    points=np.array(all_points);mid=(points.min(0)+points.max(0))/2;extent=max(np.ptp(points,axis=0))*0.72
    axes.set(xlim=(mid[0]-extent,mid[0]+extent),ylim=(mid[1]-extent,mid[1]+extent),zlim=(mid[2]-extent,mid[2]+extent),xlabel='World X (mm)',ylabel='World Y (mm)',zlabel='World Z (mm)')
    axes.view_init(elev=19,azim=-61);axes.set_box_aspect([1,1,1.15]);axes.tick_params(labelsize=8)
    axes.set_title('2  Inferred rigid geometry\nDashed = partially observed translation',loc='center',fontweight='bold',fontsize=11)
    lines=['3  Symbolic + metric state','',f"time = {record['time']:.2f} s   units = mm",'source = RGB-D   frame = world','']
    for name in ['blue','red','green']:
        obj=record['objects'][name];center=np.array(obj['center_mm']);sigma=np.array(obj['position_sigma_mm'])
        lines += [name+': rigid_cube',f'  center = {np.round(center,1).tolist()}',f'  position_sigma = {np.round(sigma,1).tolist()}',f'  translation_rank = {obj["observed_translation_rank"]}/3']
        if 'target_delta_mm' in obj:lines += ['  to_desired = '+str(np.round(obj['target_delta_mm'],1).tolist())]
        lines+=['']
    for upper,lower in [('red','blue'),('green','red')]:
        relation=scene.relation('support_candidate',upper,lower)
        lines += [f'support_candidate({upper}, {lower})',f'  = {relation["truth"]}']
    lines += ['','contact = unknown','temporal pose memory = disabled','desired state != observation']
    text.text(0,1,'\n'.join(lines),va='top',fontfamily='monospace',fontsize=10.1,linespacing=1.35)
    fig.suptitle('A real experiment checkpoint: pixels to geometry to symbolic reasoning',fontsize=16,fontweight='bold')
    fig.supxlabel('Corners are inferred rigid landmarks, including hidden corners. Uncertainty is heuristic. This state does not certify contact or task success.',fontsize=10)
    fig.savefig(output.with_suffix('.png'),dpi=180);fig.savefig(output.with_suffix('.svg'));plt.close(fig)
    return {'image':str(output.with_suffix('.png')),'sensor_files':used,'time':record['time'],'replayed_centers_match_runtime_record':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica',type=Path,required=True);parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=int,default=-1)
    args=parser.parse_args();print(json.dumps(render(args.replica,args.trial,args.output,args.checkpoint),indent=2))
