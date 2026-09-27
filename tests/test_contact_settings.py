"""Contact-solver experiments must be opt-in and preserve replay provenance."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aspire_so100.simulation_settings import contact_settings, apply_contact_settings

ROOT=Path(__file__).resolve().parents[1]


class ContactSettingsTests(unittest.TestCase):
    def test_default_and_recorded_profile_are_explicit(self):
        model=SimpleNamespace(opt=SimpleNamespace(noslip_iterations=0))
        self.assertEqual(apply_contact_settings(model),{'noslip_iterations':0})
        self.assertEqual(model.opt.noslip_iterations,0)
        self.assertEqual(apply_contact_settings(model,{'noslip_iterations':10}),{'noslip_iterations':10})
        self.assertEqual(model.opt.noslip_iterations,10)
        for value in ({'noslip_iterations':True},{'noslip_iterations':-1},
                      {'noslip_iterations':101},{'noslip_iterations':1.5},{'friction':2}):
            with self.subTest(value=value),self.assertRaises(ValueError):contact_settings(value)

    def test_replay_restores_contact_profile_and_legacy_default(self):
        from aspire_so100 import playback
        for profile in (None,{'noslip_iterations':10}):
            result={'task':'inventory_tower','seed':1,'scene':{'object_count':3,'recipe':'cube_triple','order':['blue_cube','red_cube','green_cube']}}
            if profile is not None:result['physics']=profile
            fake=SimpleNamespace(scene_metadata=result['scene'],model=SimpleNamespace(opt=SimpleNamespace(noslip_iterations=99)))
            with patch.object(playback,'RecordedInventoryTowerEnv',return_value=fake):
                restored=playback.recorded_env_for_result(result,Path('/tmp'))
            self.assertEqual(restored.model.opt.noslip_iterations,0 if profile is None else 10)

    def test_default_benchmark_rejects_nondefault_physics(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        try:
            spec=importlib.util.spec_from_file_location('columns_contact_test',ROOT/'scripts/evaluate_columns.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            sensor=module.COLUMN_SENSOR_PROFILE
            with self.assertRaisesRegex(RuntimeError,'contact settings'):
                module.validate_column_result({'sensor':sensor,'physics':{'noslip_iterations':10}}, {'sensor':sensor},'unused',Path('/tmp/result.json'))
        finally:sys.path.pop(0)


if __name__=='__main__':unittest.main()
