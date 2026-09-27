"""An unavailable unrelated object must not invalidate another object's estimate."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_held_geometry_probe import allow_initial_unknown
from build_geometry_step_session import library
from test_shared_observer_experiment import API, OBJECTS


class ProbeAPI(API):
    def camera(self, view='side'):
        # Both view requests reuse this static fixture; failure is explicitly injected.
        return super().camera()


class InitialUnknownTests(unittest.TestCase):
    def test_only_unavailable_object_is_unknown_in_the_development_patch(self):
        source=library('symbolic');patched=allow_initial_unknown(source)
        for code,tolerant in ((source,False),(patched,True)):
            namespace={};exec(compile(code,'<initial-unknown-probe>','exec'),namespace)
            backend=namespace['MixedPnPObservation'];original=backend.observe_views
            missing=OBJECTS[-1]['color'];available=OBJECTS[0]['color']
            def observe(self,views,color,model,plane):
                if color==missing:raise ValueError('Injected unavailable view')
                return original(self,views,color,model,plane)
            backend.observe_views=observe
            if tolerant:
                session=namespace['GEOMETRY'].session(ProbeAPI(),OBJECTS)
                self.assertIn('center',session.get(available))
                with self.assertRaises(ValueError):session.get(missing)
            else:
                with self.assertRaises(ValueError):namespace['GEOMETRY'].session(ProbeAPI(),OBJECTS)


if __name__=='__main__':unittest.main()
