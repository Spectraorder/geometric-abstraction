"""This assistant's single declared repair: bounded initial visibility reacquisition."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
LAB=Path(__file__).resolve().parents[1]
source=(ROOT/'policies/stack_three_blocks_visual_geometry.py').read_text()
old='''    last_frame = [None]

    def capture():
        s = api.camera()
'''
new='''    last_frame = [None]
    initial_observation = [True]

    def capture():
        s = api.camera()
        if initial_observation[0]:
            # Task color contract + sensor evidence only. No cached object coordinates.
            for observation_attempt in range(3):
                rgb_check = np.asarray(s['rgb'], float)
                rr, gg, bb = rgb_check[...,0], rgb_check[...,1], rgb_check[...,2]
                valid_check = np.asarray(s['valid_depth'], bool) & np.isfinite(s['depth_m'])
                visible = [
                    (rr>75)&(rr>1.65*gg)&(rr>1.55*bb),
                    (gg>65)&(gg>1.45*rr)&(gg>1.3*bb),
                    (bb>65)&(bb>1.55*rr)&(bb>1.2*gg)]
                if all(np.count_nonzero(mask & valid_check)>=8 for mask in visible):
                    initial_observation[0] = False
                    api.note('Initial task colors re-observed; continue with fresh RGB-D.')
                    break
                api.note('Initial geometry incomplete: hold arm, request fresh observation.')
                if observation_attempt == 2:
                    raise RuntimeError('Observation budget exhausted; no arm motion issued.')
                api.wait(.2)
                s = api.camera()
'''
assert source.count(old)==1
output=LAB/'policies/reobserve_then_stack.py'; output.write_text(source.replace(old,new))
(LAB/'policies/provenance.json').write_text(json.dumps({'author':'current assistant / gpt-6-astra','mechanism':'code repair, no model weight updates','base_sha256':hashlib.sha256(source.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'hypothesis':'Two transient image occlusions at initialization should be recoverable through bounded re-observation; persistent occlusion must stop without motion.','training_from_zero':False,'declared_before_trials':True},indent=2)+'\n')
