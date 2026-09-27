"""Render recorded sensor-only checkpoint examples, without simulator object state."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reasoning_checkpoint_state import ReasoningCheckpoints

ROOT=Path(__file__).resolve().parents[1]


def side_frame(directory, stamp):
    calls=[c for c in json.loads((directory/'api_calls.json').read_text()) if c['method']=='camera']
    files=sorted(directory.glob('camera_*.npz'))
    if len(calls)!=len(files):raise ValueError('Camera metadata mismatch')
    matches=[]
    for p,call in zip(files,calls):
        if call['params']['view']!='side':continue
        with np.load(p) as z:
            if abs(float(z['time'])-stamp)<1e-6:
                matches.append((p,z['intrinsics'].copy(),z['camera_to_world'].copy()))
    if not matches:raise ValueError('No same-time side image for checkpoint')
    p,k,transform=matches[-1]
    return p,np.asarray(Image.open(p.with_suffix('.png')).convert('RGB')),k,transform


def render():
    diagnostic=ROOT/'docs/evaluations/reasoning_checkpoint_replay_02.json'
    report=json.loads(diagnostic.read_text())
    examples=[next(r for r in report['large_changes'] if r['joint_constraint_rank']==rank) for rank in (3,0)]
    fig,axes=plt.subplots(2,3,figsize=(12,7),gridspec_kw={'width_ratios':[1.15,1.15,1]})
    provenance={'source':'Recorded pilot training sensors and fitted state only; no physics arrays, private poses, outcome labels, or reserved data.',
                'selection':'First rank-3 and first rank-0 large changes in the declared training replay.',
                'diagnostic_sha256':hashlib.sha256(diagnostic.read_bytes()).hexdigest(),'examples':[]}
    for row,event in enumerate(examples):
        path=ROOT/event['source']; records=[r for r in json.loads(path.read_text()) if r.get('kind')=='planner_state']
        indices=[int(event[key].split(':',1)[0]) for key in ('from_label','to_label')]
        before,after=[records[i] for i in indices]; name=event['object']
        memory=ReasoningCheckpoints('figure-example');memory.remember('before',before['state'],episode='figure-example')
        packet=memory.compare('before',after['state'],episode='figure-example')
        change=packet['checkpoint_changes']['objects'][name];sources=[]
        for col,record in enumerate((before,after)):
            state=record['state'];p,rgb,k,transform=side_frame(path.parent,state['time'])
            xyz=np.asarray(state['objects'][name]['center']);camera=transform[:3,:3].T@(xyz-transform[:3,3])
            if camera[2]<=0:raise ValueError('Estimated center is behind the camera')
            uv=k@camera;uv=uv[:2]/uv[2]
            ax=axes[row,col];ax.imshow(rgb);ax.plot(*uv,marker='+',markersize=15,markeredgewidth=3,color='white')
            ax.plot(*uv,marker='+',markersize=13,markeredgewidth=1.5,color='black')
            ax.set_title(('Before' if col==0 else 'After')+f": {record['label']}\nt = {state['time']:.2f} s",fontsize=10)
            height,width=rgb.shape[:2]
            ax.set_xlim(.25*width,.78*width);ax.set_ylim(.85*height,.30*height)
            ax.axis('off')
            sources.append({'camera':str(p.relative_to(ROOT)),'camera_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                            'image_sha256':hashlib.sha256(p.with_suffix('.png').read_bytes()).hexdigest(),
                            'projected_center_uv':uv.tolist()})
        ax=axes[row,2];delta=1000*np.asarray(change['estimated_translation_m']);positions=np.arange(3)
        ax.bar(positions-.16,delta,width=.32,color='#9b9b9b',label='Raw estimate difference')
        projected=change['jointly_constrained_translation_m']
        if projected is not None:
            ax.bar(positions+.16,1000*np.asarray(projected),width=.32,color='#267db3',label='Jointly constrained part')
        else:
            ax.text(.5,.83,'Joint component: unknown\nNo common constrained direction',ha='center',va='center',transform=ax.transAxes,fontsize=9,color='#9b521c')
        ax.axhline(0,color='#444444',lw=.6);ax.set_xticks(positions,['X','Y','Z'])
        ax.set_ylabel('Estimated translation (mm)');ax.grid(axis='y',alpha=.2)
        ax.set_title(f"{name} {after['state']['objects'][name]['shape'].replace('_',' ')}\nJoint constraint rank: {change['joint_constraint_rank']}/3",fontsize=10)
        ax.legend(fontsize=7,loc='lower left')
        provenance['examples'].append({'event':event,'change':change,'sensor_frames':sources})
    fig.suptitle('Action checkpoints: estimated motion and jointly constrained directions',fontsize=13)
    fig.text(.5,.015,'Cropped side RGB; crosses mark estimated centers. Fitter constraints do not certify physical motion or calibrated confidence.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.95))
    target=ROOT/'docs/media/checkpoint_constraints.png';fig.savefig(target,dpi=160);plt.close(fig)
    provenance['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/'docs/evaluations/checkpoint_constraints_figure.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(target)


if __name__=='__main__':render()
