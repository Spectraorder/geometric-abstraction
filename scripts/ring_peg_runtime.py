"""Permitted sensor/robot interface; private task evaluation is outside this class."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ring_peg_geometry import inventory_table, observe_ring, observe_peg, initialized_model, register_ring
from ring_peg_robot import PadActuator


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value


class RingSensors:
    def __init__(self,env,output):
        self.env,self.output=env,Path(output)
        self.robot=PadActuator(env)
        self.height=None
        self.model=None
        self.frames=0
        self.colors={key:env.layout[key+'_color'] for key in ('ring','peg')}
        self.records=[]
        self.table=None

    def camera(self,view='side'):
        if view not in ('side','wrist'): raise ValueError('Unknown view')
        frame=self.env.get_overhead_rgbd() if view=='side' else self.env.get_wrist_rgb(depth=True)
        stem=self.output/f'camera_{self.frames:03d}_{view}'
        np.savez_compressed(stem.with_suffix('.npz'),**frame)
        Image.fromarray(frame['rgb']).save(stem.with_suffix('.png'))
        self.frames+=1
        return frame

    def observe(self,name):
        if name not in self.colors: raise ValueError('Unknown public object')
        errors=[]
        for view in ('side','wrist'):
            frame=self.camera(view)
            try:
                if self.table is None: self.table=inventory_table(frame)
                if name=='ring':
                    try:
                        record=observe_ring(frame,self.colors[name],self.table,self.height)
                    except ValueError:
                        if self.model is None: raise
                        record=register_ring(frame,self.colors[name],self.table,self.model)
                    if self.model is None: self.model=initialized_model(record)
                    self.height=record['height_m']
                else: record=observe_peg(frame,self.colors[name],self.table)
                record.update(object=name,view=view,table_plane=self.table.tolist())
                self.records.append(record)
                return record
            except (ValueError,RuntimeError) as error: errors.append(view+': '+str(error))
        raise ValueError('No current geometry: '+'; '.join(errors))

    def task(self):
        return {'goal':'Place the movable ring over the fixed upright peg and release it.',
                'objects':[{'id':k,'color':v} for k,v in self.colors.items()],
                'assumptions':'One rigid perforated extrusion initially flat on the table; one fixed upright cylindrical peg. No metric object dimensions supplied.'}

    def dispatch(self,method,params):
        if method=='camera': return self.camera(**params)
        if method=='task': return self.task()
        if method=='observe': return self.observe(**params)
        if method=='robot': return self.robot.current()
        if method=='plan_grasp': return self.robot.plan_grasp(**params)
        if method=='plan_pad': return self.robot.plan_pad(**params)
        if method=='move_pad': return self.robot.move_pad(**params)
        if method=='gripper':
            angle,seconds=float(params['angle']),float(params.get('seconds',.6))
            if not -.174<=angle<=1.75 or not .1<=seconds<=10: raise ValueError('Invalid jaw command')
            self.env.set_gripper(angle,seconds);return self.robot.current()
        if method=='wait':
            seconds=float(params['seconds'])
            if not 0<=seconds<=10: raise ValueError('Invalid wait')
            self.env.wait(seconds);return None
        if method=='note':
            self.records.append({'note':str(params['text'])[:2000],'time':float(self.env.data.time)});return None
        raise ValueError('Unknown public method')
