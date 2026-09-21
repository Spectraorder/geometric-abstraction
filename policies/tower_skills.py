"""Three-block tower (blue base -> small red -> green cube) with verified skills.

The tower is expressed as a plan and built by the plan executor, which
re-perceives every support live, verifies each placement, and repairs from
the lowest broken level if a structure check fails.
"""

from aspire_so100.tower_plan import execute_plan

PLAN = {
    "name": "blue_red_green",
    "placements": [
        {"block": "blue_block"},
        {"block": "small_red_block", "support": 0},
        {"block": "green_block", "support": 1},
    ],
}


def run(api) -> None:
    execute_plan(api, PLAN, check=api.check_structure)
