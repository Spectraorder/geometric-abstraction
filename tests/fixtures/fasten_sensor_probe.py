"""Infrastructure probe only; never used as a learned/best policy."""
def run(api):
    goal=api.task()
    frame=api.camera()
    robot=api.proprioception()
    api.note('Probe: '+goal['kind']+' RGB-D '+str(frame['rgb'].shape))
    api.gripper(.8,.2)
    api.wait(.2)
