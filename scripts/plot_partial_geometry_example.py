"""Plot real RGB-D evidence and conditional center intervals, never object truth."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def render(output):
    fixture=ROOT/'tests/fixtures/partial_cuboid_failure_rgbd.npz'
    evidence=ROOT/'docs/evaluations/observer_input_failure_replay.json'
    report=json.loads(evidence.read_text())
    exact=report['variants'][0]
    if exact['status']!='rejected' or exact['model']!='current' or exact['mask']!='robot_mask':
        raise ValueError('Require the exact rejected current input')
    fit=exact['diagnostic_bounds'];comparison=report['conditional_prediction_comparison']
    if comparison is None or comparison['pose_fused']: raise ValueError('Require an unfused conditional prediction')
    bounds=np.asarray(fit['finite_center_bounds_m']);mid=bounds.mean(axis=1)
    prediction=np.asarray(comparison['prediction_in_fitted_basis_m'])
    offsets=(bounds-mid[:,None])*1000; predicted_offset=(prediction-mid)*1000
    with np.load(fixture) as data:
        side=data['side_rgb'];wrist=data['wrist_rgb']
        stamps=[float(data[v+'_time']) for v in ('side','wrist')]
    if any(abs(t-report['time'])>1e-6 for t in stamps): raise ValueError('Image timestamps differ from constraint evidence')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13,8.7),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1,.92])
    for view,rgb,pos in [('Side camera',side,grid[0,0]),('Wrist camera',wrist,grid[0,1])]:
        ax=fig.add_subplot(pos);ax.imshow(rgb);ax.set_axis_off();ax.set_title(view+' · actual RGB at 75.9 s',loc='left',fontweight='bold')
    ax=fig.add_subplot(grid[1,0]);basis=np.asarray(fit['rotation']).T
    for index in range(3):
        color='#167b80' if fit['constrained_axes'][index] else '#c08525'
        ax.plot(offsets[index],[index,index],color=color,lw=13,alpha=.5,solid_capstyle='butt')
        ax.plot(offsets[index],[index,index],'|',color=color,markersize=18,markeredgewidth=2)
        ax.plot(predicted_offset[index],index,'D',color='#873b9c',markersize=7,zorder=4)
        ax.annotate(f'{(bounds[index,1]-bounds[index,0])*1000:.3f} mm wide',
            (offsets[index,0],index),xytext=(0,14),textcoords='offset points',fontsize=9)
    ax.set_yticks(range(3),[f'Axis {i+1}: '+('face constrained' if flag else 'hidden direction')
                         for i,flag in enumerate(fit['constrained_axes'])])
    ax.set_ylim(2.55,-.65);ax.set_xlim(-2.25,4.05);ax.axvline(0,color='#aaaaaa',lw=.7,zorder=0)
    ax.grid(axis='x',alpha=.2);ax.set_xlabel('Center coordinate relative to each interval midpoint (mm)\nMidpoints are display references, not measured centers',fontsize=9)
    ax.set_title('Current geometric constraints versus a motion hypothesis',loc='left',fontsize=11,fontweight='bold',pad=14)
    ax=fig.add_subplot(grid[1,1]);ax.set_axis_off()
    lines=[('TARGET','Red thin prism · training layout 990205'),
           ('VISUAL RESULT','Point pose unavailable; two face directions constrained.'),
           ('WHY IT STOPS','Hidden interval: 1.973 mm > existing 1.5 mm gate.'),
           ('MODEL CONDITION','Bounds assume fitted orientation and visual dimensions.\nThey are not calibrated confidence intervals.'),
           ('PREDICTION CHECK','Purple diamonds: rigid-attachment prediction, not observation.\nOutside current bounds by 2.834 mm and 0.307 mm\nalong two directions; no fusion or pose recovery.'),
           ('AGENT FEEDBACK','Keep one target and its current constraints.\nDo not replace an unknown center with a plausible point.')]
    y=.98
    for title,body in lines:
        ax.text(0,y,title,color='#167b80',fontweight='bold',fontsize=9,va='top')
        ax.text(0,y-.046,body,fontsize=10,va='top',linespacing=1.4)
        y-=.14 if '\n' not in body else (.18 if body.count('\n')==1 else .22)
    fig.suptitle('From recorded simulation views to a compact geometry abstraction',fontsize=17,fontweight='bold')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160,facecolor='white');plt.close(fig)
    metadata={'scope':__doc__,'seed':report['seed'],'time':report['time'],'object':report['object'],
              'source_sha256':{str(p):sha(p) for p in (Path(__file__),fixture,evidence)},
              'fitted_basis_rows_world':basis.tolist(),'center_bounds_m':bounds.tolist(),
              'interval_widths_mm':((bounds[:,1]-bounds[:,0])*1000).tolist(),
              'prediction_relative_to_interval_midpoint_mm':predicted_offset.tolist(),
              'limitations':['RGB images and geometric evidence are actual development observations.',
                             'Interval widths and predictions are model-conditioned, not ground truth.',
                             'This illustrates evidence; no task success or pose accuracy gain is claimed.']}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    render(parser.parse_args().output.resolve())
