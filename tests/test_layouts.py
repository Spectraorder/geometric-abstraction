"""Layout reproducibility and physical separation for randomized experiments."""
import unittest
import numpy as np
from aspire_so100.layouts import BLOCK_HALF_SIZES, RANDOM_XY_BOUNDS, random_block_positions


class LayoutTests(unittest.TestCase):
    def test_random_layouts_repeat_and_have_clearance(self):
        for seed in [0, 1, 7, *range(101, 106), *range(201, 206)]:
            positions = random_block_positions(seed)
            repeated = random_block_positions(seed)
            for name, xy in positions.items():
                np.testing.assert_array_equal(xy, repeated[name])
                for value, (low, high) in zip(xy, RANDOM_XY_BOUNDS):
                    self.assertTrue(low <= value <= high)
            names = list(positions)
            for i, first in enumerate(names):
                for second in names[i + 1:]:
                    required = (np.array(BLOCK_HALF_SIZES[first][:2]) +
                                np.array(BLOCK_HALF_SIZES[second][:2]) + .025)
                    self.assertTrue(np.any(np.abs(positions[first] - positions[second]) >= required))

    def test_colors_can_exchange_sides(self):
        layouts = [random_block_positions(seed) for seed in range(101, 106)]
        for name in BLOCK_HALF_SIZES:
            xs = [layout[name][0] for layout in layouts]
            self.assertLess(min(xs), 0)
            self.assertGreater(max(xs), 0)
