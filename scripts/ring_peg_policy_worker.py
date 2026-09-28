"""Isolated policy client; only explicit public RPC results enter this process."""
import json
import sys
import traceback


class API:
    def request(self,method,**params):
        print(json.dumps({'method':method,'params':params}),flush=True)
        response=json.loads(sys.stdin.readline())
        if 'error' in response:raise RuntimeError(response['error'])
        return response['result']
    def task(self):return self.request('task')
    def observe(self,name):return self.request('observe',name=name)
    def robot(self):return self.request('robot')
    def plan_grasp(self,fixed_pad_m,closing_direction,jaw):
        return self.request('plan_grasp',fixed_pad_m=list(fixed_pad_m),closing_direction=list(closing_direction),jaw=float(jaw))
    def plan_pad(self,fixed_pad_m,roll,jaw):
        return self.request('plan_pad',fixed_pad_m=list(fixed_pad_m),roll=float(roll),jaw=float(jaw))
    def move_pad(self,point,roll,seconds=1.5):
        return self.request('move_pad',point=list(point),roll=float(roll),seconds=float(seconds))
    def gripper(self,angle,seconds=.6):return self.request('gripper',angle=float(angle),seconds=float(seconds))
    def wait(self,seconds):return self.request('wait',seconds=float(seconds))
    def note(self,text):return self.request('note',text=str(text)[:2000])


try:
    namespace={}
    exec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec'),namespace)
    namespace['run'](API())
    print(json.dumps({'method':'done','params':{}}),flush=True)
except BaseException as error:
    print(json.dumps({'method':'failed','params':{'error':type(error).__name__+': '+str(error)}}),flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
