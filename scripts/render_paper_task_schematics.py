#!/usr/bin/env python3
"""Render illustrative task stages, not policy executions or evaluation evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import math

import mujoco
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from aspire_so100 import inventory_structure
from aspire_so100.stacking_env import TABLE_HEIGHT
from run_three_cube_trial import cube_recipe
from online_bridge_scene import OnlineBridgeTowerEnv
from online_relocation_scene import RelocationEnv
from ring_peg_scene import RingPegEnv

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'docs/evaluations/task_illustrations'
WIDTH, HEIGHT = 1200, 900
FLAT = [math.sqrt(.5), 0., math.sqrt(.5), 0.]
UPRIGHT = [1., 0., 0., 0.]


def pose(model, data, name, xyz, quat=UPRIGHT):
    joint = model.joint(f'{name}/{name}_joint')
    address = int(joint.qposadr[0])
    data.qpos[address:address+7] = [*xyz, *quat]


def prepare_model(model):
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    model.vis.quality.offsamples = 4
    model.vis.quality.shadowsize = 4096
    # Display-only closeups omit the robot; physics, task geometry and colors stay intact.
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body:
            name = model.body(body).name
            if name and (name == 'robot' or name.startswith('robot/')):
                model.geom_group[geom] = 5
                break
            body = int(model.body_parentid[body])


def frame(model, data, renderer, camera):
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[5] = 0
    option.sitegroup[:] = 0
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render().copy()


def camera_for(model, data, states, names):
    points = []
    original = data.qpos.copy()
    for state in states:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        for name in names:
            body = model.body(f'{name}/{name}').id
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] != body:
                    continue
                radius = float(model.geom_rbound[gid])
                center = data.geom_xpos[gid]
                points.extend([center-radius, center+radius])
    data.qpos[:] = original
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (low+high)/2
    camera.distance = max(.20, 1.18 * float(np.linalg.norm(high-low)))
    camera.azimuth = 105
    camera.elevation = -48
    return camera


def structure_states(env, kind):
    model, data = env.model, env.data
    mujoco.mj_resetDataKeyframe(model, data, 0)
    nodes = list(env.recipe.nodes)
    unit = 2*model.geom(f'{nodes[0].object_name}/{nodes[0].object_name}_geom').size[2]
    origin = np.array([0., -.30])
    goals, heights, rotations = {}, {}, {}
    for node in nodes:
        name = node.object_name
        geom = model.geom(f'{name}/{name}_geom')
        rotations[name] = FLAT if node.orientation == 'flat_x' else UPRIGHT
        heights[node.key] = 2*geom.size[1 if name.endswith('cylinder') else 0 if node.orientation == 'flat_x' else 2]
        bottom = max((goals[parent][2] + heights[parent]/2 for parent in node.supports), default=TABLE_HEIGHT)
        goals[node.key] = [*(origin+unit*np.asarray(node.grid)), bottom+heights[node.key]/2]
    if kind == 'relocation':
        by_key = {n.key:n for n in nodes}
        left, right = [-.09,-.30], [.09,-.30]
        for key, level in [('source',0),('destination',0),('middle',1),('upper',2)]:
            node = by_key[key]
            xy = right if key == 'destination' else left
            pose(model,data,node.object_name,[*xy,TABLE_HEIGHT+(level+.5)*unit])
        states = [data.qpos.copy()]
        # An illustrative temporary placement frees the middle cube.
        pose(model,data,by_key['upper'].object_name,[0.,-.39,TABLE_HEIGHT+.5*unit])
        pose(model,data,by_key['middle'].object_name,[*right,TABLE_HEIGHT+1.5*unit])
        states.append(data.qpos.copy())
        pose(model,data,by_key['upper'].object_name,[*right,TABLE_HEIGHT+2.5*unit])
        states.append(data.qpos.copy())
        return states
    positions = [(-.105,-.34),(.105,-.34),(-.06,-.24),(.09,-.24),(0.,-.39)]
    for node,xy in zip(nodes,positions):
        pose(model,data,node.object_name,[*xy,TABLE_HEIGHT+heights[node.key]/2],rotations[node.object_name])
    states = [data.qpos.copy()]
    count = 2 if kind == 'stacking' else 3
    for index,node in enumerate(nodes):
        pose(model,data,node.object_name,goals[node.key],rotations[node.object_name])
        if index+1 == count:
            states.append(data.qpos.copy())
    states.append(data.qpos.copy())
    return states


def ring_states(env, online=False):
    model,data=env.model,env.data
    mujoco.mj_resetDataKeyframe(model,data,0)
    mujoco.mj_forward(model,data)
    g=env.geometry
    peg=data.xpos[model.body('peg/peg').id].copy()
    table=TABLE_HEIGHT
    xy=np.asarray(env.layout['ring_xy'])
    stages=[[*xy,table+g.height/2], [*peg[:2],table+g.peg_height+.025],
            [*peg[:2],table+g.height/2]]
    if online:
        # The first panel depicts a possible state after the shared grasp/lift prefix.
        stages[0][2]=table+.085
    states=[]
    for xyz in stages:
        pose(model,data,'ring',xyz)
        states.append(data.qpos.copy())
    return states


def strip(images, title, labels, output, subtitle):
    fig=plt.figure(figsize=(12,3.8),facecolor='white')
    for index,(image,label) in enumerate(zip(images,labels)):
        ax=fig.add_axes([.015+index*.333,.18,.302,.69])
        ax.imshow(image);ax.axis('off')
        ax.set_title(label,fontsize=11,pad=7)
        if index<2:
            fig.add_artist(FancyArrowPatch((.322+index*.333,.51),(.343+index*.333,.51),
                           transform=fig.transFigure,arrowstyle='-|>',mutation_scale=14,color='#63758A',lw=1.1))
    fig.suptitle(title,fontsize=15,y=.995)
    fig.text(.5,.065,subtitle,ha='center',fontsize=9,color='#455466')
    fig.text(.5,.013,'Illustrative scene configurations; not recorded policy trajectories. Robot omitted; framing adjusted per panel.',
             ha='center',fontsize=8,color='#687584')
    fig.savefig(output.with_suffix('.pdf'),metadata={'CreationDate':None})
    fig.savefig(output.with_suffix('.png'),dpi=300)
    plt.close(fig)


def gallery(records, mode):
    selected=[r for r in records if r['mode']==mode]
    fig,axes=plt.subplots(3,3,figsize=(11.5,9.0))
    fig.subplots_adjust(left=.015,right=.985,top=.91,bottom=.035,hspace=.28,wspace=.045)
    for row,record in enumerate(selected):
        for col,file in enumerate(record['frames']):
            ax=axes[row,col];ax.imshow(Image.open(OUT/file));ax.axis('off')
            ax.set_title(record['labels'][col],fontsize=10,pad=4)
        axes[row,1].text(.5,1.18,record['title'],transform=axes[row,1].transAxes,
                        ha='center',fontsize=12,fontweight='bold')
    fig.text(.5,.01,'Illustrative configurations using the experiment scene assets; no policy execution or success claim.',
             ha='center',fontsize=8,color='#687584')
    for suffix in ('pdf','png'):
        kwargs={'metadata':{'CreationDate':None}} if suffix=='pdf' else {}
        fig.savefig(OUT/f'{mode}_tasks.{suffix}',dpi=300,**kwargs)
    fig.savefig(Path('/tmp')/f'{mode}_tasks_preview.png',dpi=90)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42})
    definitions=[
        ('single_bridge','single_policy','Bridge + cylinder',['Separate objects','Span two foundations','Place the cylinder'],
         'Build a bridge and place a cylinder on the beam.'),
        ('single_stacking','single_policy','Three-cube stacking',['Separate cubes','Place the middle cube','Complete the ordered stack'],
         'Build an ordered three-cube tower.'),
        ('single_ring','single_policy','Ring over peg',['Ring beside the peg','Align the opening','Lower and release'],
         'Grasp material around the opening, align with the fixed peg, and release.'),
        ('online_tower','online','Bridge-tower construction',['Separate objects','Construct the bridge','Stack above the beam'],
         'Compose bridge and stacking skills in one continuing scene.'),
        ('online_ring','online','Online ring placement',['After shared grasp / lift','Observe and align','Release around the peg'],
         'Use the shared lift skill, then choose alignment, release, and recovery online.'),
        ('online_relocation','online','Tower relocation',['Source stack + destination','Use a temporary placement','Rebuild on the destination'],
         'Transfer the upper two cubes in order; preserve or restore both foundations.'),
    ]
    records=[]
    with TemporaryDirectory(prefix='paper-task-schematics-') as temp:
        for key,mode,title,labels,subtitle in definitions:
            folder=Path(temp)/key
            if key=='single_bridge':
                env=inventory_structure.InventoryStructureEnv(folder,scene_seed=1,object_count=4,recipe='thin_beam_bridge')
            elif key=='single_stacking':
                with patch.object(inventory_structure,'structure_recipe',cube_recipe):
                    env=inventory_structure.InventoryStructureEnv(folder,scene_seed=1,object_count=3,recipe='cube_triple')
            elif key.endswith('ring'):
                env=RingPegEnv(folder,scene_seed=1)
            elif key=='online_tower':
                env=OnlineBridgeTowerEnv(folder,scene_seed=1)
            else:
                env=RelocationEnv(folder,scene_seed=1)
            renderer=None
            try:
                if key.endswith('ring'):
                    states=ring_states(env,online=key=='online_ring');names=['ring','peg']
                else:
                    kind='relocation' if key.endswith('relocation') else 'stacking' if key.endswith('stacking') else 'bridge'
                    states=structure_states(env,kind);names=list(env.order)
                prepare_model(env.model)
                renderer=mujoco.Renderer(env.model,width=WIDTH,height=HEIGHT)
                images=[];files=[];cameras=[]
                for index,state in enumerate(states):
                    env.data.qpos[:]=state
                    camera=camera_for(env.model,env.data,[state],names)
                    cameras.append(dict(lookat=camera.lookat.tolist(),distance=camera.distance,
                                        azimuth=camera.azimuth,elevation=camera.elevation))
                    image=frame(env.model,env.data,renderer,camera)
                    name=f'{key}_{index:02d}.png';Image.fromarray(image).save(OUT/name)
                    images.append(image);files.append(name)
                strip(images,title,labels,OUT/key,subtitle)
                records.append(dict(key=key,mode=mode,title=title,labels=labels,description=subtitle,
                    frames=files,objects=names,qpos=[state.tolist() for state in states],
                    cameras=cameras,
                    frame_width=WIDTH,frame_height=HEIGHT,scene_seed_for_asset_identity=1))
                print(key,'rendered',flush=True)
            finally:
                if renderer is not None:renderer.close()
                env.close()
    for mode in ('single_policy','online'):gallery(records,mode)
    sources=[Path(__file__),ROOT/'src/aspire_so100/inventory_structure.py',ROOT/'src/aspire_so100/structure_tasks.py',
             ROOT/'scripts/run_three_cube_trial.py',ROOT/'scripts/online_bridge_scene.py',
             ROOT/'scripts/online_relocation_scene.py',ROOT/'scripts/ring_peg_scene.py']
    rendered_names = {name for r in records for name in r['frames']}
    rendered_names.update(r['key']+suffix for r in records for suffix in ('.pdf', '.png'))
    rendered_names.update(mode+'_tasks'+suffix for mode in ('single_policy', 'online')
                          for suffix in ('.pdf', '.png'))
    manifest=dict(kind='constructed_task_schematics',records=records,
        interpretation='Illustrative poses, not measured experimental trajectories or evidence of success. No policies, model requests, or physics rollouts were executed.',
        changes='Only temporary visualization models and object poses; robot render group hidden for closeups.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name in rendered_names})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
