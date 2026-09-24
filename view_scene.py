import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from mjviser import Viewer

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from aspire_so100.layouts import CUBE_HALF_SIZE
from aspire_so100.block_catalog import BLOCK_COLORS, SHAPE_DIMENSIONS, CATALOG_POSITIONS, generate_block_layout


PROJECT_ROOT = Path(__file__).resolve().parent
SO100_XML = PROJECT_ROOT / "assets" / "trs_so_arm100" / "so_arm100.xml"

TABLE_HEIGHT = 0.40


def get_so100_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SO100_XML))
    # Look forward past the wrist toward the jaws and grasped object.
    spec.body("Fixed_Jaw").add_camera(
        name="wrist_camera",
        pos=(0.0, -0.015, 0.045),
        xyaxes=(-1.0, 0.0, 0.0, 0.0, -0.541, 0.841),
        fovy=75.0,
    )
    # Cartesian primitives and trace overlays use this grasp-center site.
    spec.body("Fixed_Jaw").add_site(
        name="ee_site",
        # Offset toward the moving jaw so the site is centered in an open grip.
        pos=(-0.020, -0.085, 0.0),
        size=(0.004,),
        rgba=(0.1, 0.9, 0.2, 0.7),
    )
    return spec


def get_table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()

    body = spec.worldbody.add_body(name="table")
    # Raised mast on the right edge, looking diagonally toward the workspace.
    horizontal = math.hypot(.48, .30)
    length = math.hypot(horizontal, .65)
    body.add_camera(
        name="overhead_camera",
        pos=(0.48, -0.30, 0.725),
        xyaxes=(.30 / horizontal, .48 / horizontal, 0.0,
                -.65 * .48 / (length * horizontal),
                .65 * .30 / (length * horizontal), horizontal / length),
        fovy=60.0,
    )
    body.add_geom(name="camera_mast", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(.51,-.30,.35), size=(.012,.375), rgba=(.15,.17,.19,1),
                  contype=0, conaffinity=0, mass=0)
    body.add_geom(name="camera_mount", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=(.51,-.30,.725), size=(.025,.035,.022), rgba=(.12,.12,.12,1),
                  contype=0, conaffinity=0, mass=0)
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        # MuJoCo uses half-sizes: width, depth, thickness.
        size=(0.40, 0.35, 0.025),
        rgba=(0.55, 0.38, 0.22, 1.0),
        friction=(1.0, 0.005, 0.0001),
    )

    return spec


def get_block_spec(
    name: str,
    half_size: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()

    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half_size,
        mass=0.03,
        rgba=color,
        friction=(1.0, 0.005, 0.0001),
    )

    return spec


def get_cylinder_spec(
    name: str,
    radius: float,
    half_height: float,
    color: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    """Create a graspable cylinder whose axis points upward."""
    spec = mujoco.MjSpec()

    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        # Cylinder sizes are radius and half-height.
        size=(radius, half_height),
        mass=0.03,
        rgba=color,
        friction=(1.0, 0.005, 0.0001),
    )

    return spec


def get_triangular_prism_spec(
    name: str,
    side: float,
    color: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    """Half a cube cut along an XZ diagonal, with a flat rectangular bottom.

    Six vertices form two right-isosceles triangles extruded along Y.
    The convex mesh is both the visible surface and the collision geometry.
    """
    if not math.isfinite(side) or side <= 0:
        raise ValueError('The source cube side must be positive and finite')
    h = side / 2
    spec = mujoco.MjSpec()
    vertices = (-h, -h, -h, h, -h, -h, -h, -h, h,
                -h, h, -h, h, h, -h, -h, h, h)
    faces = (0, 1, 2, 3, 5, 4,              # triangular ends
             0, 3, 4, 0, 4, 1,              # rectangular bottom
             0, 2, 5, 0, 5, 3,              # vertical rectangle
             1, 4, 5, 1, 5, 2)              # sloping rectangle
    mesh_name = f'{name}_mesh'
    spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f'{name}_joint')
    body.add_geom(name=f'{name}_geom', type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname=mesh_name, mass=.015, rgba=color,
                  friction=(1., .005, .0001))
    return spec


def get_tetrahedron_spec(
    name: str,
    half_width: float,
    half_depth: float,
    height: float,
    color: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    """Create a tetrahedron with a triangular base and four triangular faces."""
    spec = mujoco.MjSpec()

    vertices = (
        -half_width, -half_depth, -height / 2,
        half_width, -half_depth, -height / 2,
        0.0, half_depth, -height / 2,
        0.0, 0.0, height / 2,
    )
    faces = (
        0, 2, 1,  # base
        0, 1, 3,
        1, 2, 3,
        2, 0, 3,
    )
    mesh_name = f"{name}_mesh"
    spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)

    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        mass=0.03,
        rgba=color,
        friction=(1.0, 0.005, 0.0001),
    )

    return spec


def get_arch_bridge_spec(
    name: str,
    half_depth: float,
    color: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    """Create a low, smooth arch bridge with stable compound collisions."""
    spec = mujoco.MjSpec()

    outer_half_width = 0.050
    bottom = -0.030
    top = 0.030
    inner_radius = 0.024
    spring_height = -0.010
    outer_radius = top - spring_height
    num_segments = 24

    # Build one concave visual mesh. It contains the two legs and the material
    # above the semicircular doorway, so the opening is genuinely visible.
    vertices: list[float] = []
    faces: list[int] = []
    vertex_indices: dict[tuple[float, float, float], int] = {}

    def vertex(point: tuple[float, float, float]) -> int:
        if point not in vertex_indices:
            vertex_indices[point] = len(vertices) // 3
            vertices.extend(point)
        return vertex_indices[point]

    def quad(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        d: tuple[float, float, float],
    ) -> None:
        ia, ib, ic, id_ = (vertex(point) for point in (a, b, c, d))
        faces.extend((ia, ib, ic, ia, ic, id_))

    arch_points = []
    for index in range(num_segments + 1):
        angle = math.pi - math.pi * index / num_segments
        arch_points.append(
            (
                inner_radius * math.cos(angle),
                spring_height + inner_radius * math.sin(angle),
            )
        )

    # Front and back faces: two legs plus narrow strips over the curved opening.
    for y, front in ((-half_depth, True), (half_depth, False)):
        surface_quads = [
            (
                (-outer_half_width, y, bottom),
                (-inner_radius, y, bottom),
                (-inner_radius, y, top),
                (-outer_half_width, y, top),
            ),
            (
                (inner_radius, y, bottom),
                (outer_half_width, y, bottom),
                (outer_half_width, y, top),
                (inner_radius, y, top),
            ),
        ]
        for (x0, z0), (x1, z1) in zip(arch_points, arch_points[1:]):
            surface_quads.append(
                ((x0, y, z0), (x1, y, z1), (x1, y, top), (x0, y, top))
            )
        for corners in surface_quads:
            if front:
                quad(*corners)
            else:
                quad(corners[0], corners[3], corners[2], corners[1])

    # Outer perimeter, the bottoms of both legs, and the inside of the arch.
    quad(
        (-outer_half_width, -half_depth, bottom),
        (-outer_half_width, -half_depth, top),
        (-outer_half_width, half_depth, top),
        (-outer_half_width, half_depth, bottom),
    )
    quad(
        (outer_half_width, -half_depth, bottom),
        (outer_half_width, half_depth, bottom),
        (outer_half_width, half_depth, top),
        (outer_half_width, -half_depth, top),
    )
    quad(
        (-outer_half_width, -half_depth, top),
        (outer_half_width, -half_depth, top),
        (outer_half_width, half_depth, top),
        (-outer_half_width, half_depth, top),
    )
    for x0, x1 in ((-outer_half_width, -inner_radius), (inner_radius, outer_half_width)):
        quad(
            (x0, -half_depth, bottom),
            (x0, half_depth, bottom),
            (x1, half_depth, bottom),
            (x1, -half_depth, bottom),
        )
    quad(
        (-inner_radius, -half_depth, bottom),
        (-inner_radius, half_depth, bottom),
        (-inner_radius, half_depth, spring_height),
        (-inner_radius, -half_depth, spring_height),
    )
    quad(
        (inner_radius, -half_depth, bottom),
        (inner_radius, -half_depth, spring_height),
        (inner_radius, half_depth, spring_height),
        (inner_radius, half_depth, bottom),
    )
    for (x0, z0), (x1, z1) in zip(arch_points, arch_points[1:]):
        quad(
            (x0, -half_depth, z0),
            (x0, half_depth, z0),
            (x1, half_depth, z1),
            (x1, -half_depth, z1),
        )

    mesh_name = f"{name}_mesh"
    spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)

    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        contype=0,
        conaffinity=0,
        mass=0.0,
        rgba=color,
    )

    middle_radius = (inner_radius + outer_radius) / 2
    ring_half_thickness = (outer_radius - inner_radius) / 2
    pillar_half_height = (spring_height - bottom) / 2
    pillar_half_width = (outer_half_width - inner_radius) / 2
    geom_mass = 0.04 / 11
    collision_color = (0.0, 0.0, 0.0, 0.0)

    # Invisible compound geoms keep collision handling stable while preserving
    # the doorway instead of replacing the concave mesh with its convex hull.
    pillar_center = (outer_half_width + inner_radius) / 2
    for side, x in (("left", -pillar_center), ("right", pillar_center)):
        body.add_geom(
            name=f"{name}_{side}_pillar",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0.0, (bottom + spring_height) / 2),
            size=(pillar_half_width, half_depth, pillar_half_height),
            mass=geom_mass,
            rgba=collision_color,
            friction=(1.0, 0.005, 0.0001),
        )

    segment_half_length = middle_radius * math.pi / num_segments * 0.58
    for index in range(num_segments):
        angle = math.pi * (index + 0.5) / num_segments
        x = middle_radius * math.cos(angle)
        z = spring_height + middle_radius * math.sin(angle)
        body.add_geom(
            name=f"{name}_arch_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0.0, z),
            axisangle=(0.0, 1.0, 0.0, math.pi / 2 - angle),
            size=(segment_half_length, half_depth, ring_half_thickness),
            mass=geom_mass,
            rgba=collision_color,
            friction=(1.0, 0.005, 0.0001),
        )

    return spec


def get_hemisphere_spec(
    name: str,
    radius: float,
    color: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    """Create a closed, flat-bottomed hemisphere mesh."""
    spec = mujoco.MjSpec()

    num_rings = 5
    num_segments = 20
    vertices = [0.0, 0.0, radius / 2]  # dome apex
    for ring in range(1, num_rings + 1):
        polar_angle = math.pi * ring / (2 * num_rings)
        ring_radius = radius * math.sin(polar_angle)
        z = radius * math.cos(polar_angle) - radius / 2
        for segment in range(num_segments):
            angle = 2 * math.pi * segment / num_segments
            vertices.extend(
                (ring_radius * math.cos(angle), ring_radius * math.sin(angle), z)
            )

    base_center_index = 1 + num_rings * num_segments
    vertices.extend((0.0, 0.0, -radius / 2))

    faces: list[int] = []
    for segment in range(num_segments):
        next_segment = (segment + 1) % num_segments
        faces.extend((0, 1 + segment, 1 + next_segment))

    for ring in range(num_rings - 1):
        current = 1 + ring * num_segments
        below = current + num_segments
        for segment in range(num_segments):
            next_segment = (segment + 1) % num_segments
            faces.extend((current + segment, below + segment, below + next_segment))
            faces.extend((current + segment, below + next_segment, current + next_segment))

    base_ring = 1 + (num_rings - 1) * num_segments
    for segment in range(num_segments):
        next_segment = (segment + 1) % num_segments
        faces.extend((base_center_index, base_ring + next_segment, base_ring + segment))

    mesh_name = f"{name}_mesh"
    spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)

    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        mass=0.025,
        rgba=color,
        friction=(1.0, 0.005, 0.0001),
    )

    return spec


def make_block_cfg(
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: get_block_spec(name, half_size, color),
        init_state=EntityCfg.InitialStateCfg(
            pos=position,
            joint_pos={},
        ),
    )


def make_shape_cfg(
    spec_fn: Callable[[], mujoco.MjSpec],
    position: tuple[float, float, float],
) -> EntityCfg:
    """Wrap any free-joint object spec as an mjlab scene entity."""
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=position,
            joint_pos={},
        ),
    )


robot_cfg = EntityCfg(
    spec_fn=get_so100_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, TABLE_HEIGHT),
        joint_pos=None,  # Use the SO-100 home keyframe.
    ),
)

table_cfg = EntityCfg(
    spec_fn=get_table_spec,
    init_state=EntityCfg.InitialStateCfg(
        # Tabletop center. Its upper surface is at TABLE_HEIGHT.
        pos=(0.0, -0.30, TABLE_HEIGHT - 0.025),
        joint_pos={},
    ),
)

# Keep the original object names/configuration for historical task rollouts.
legacy_entities = {
    "robot": robot_cfg,
    "table": table_cfg,
    "red_block": make_block_cfg(
        name="red_block",
        position=(-0.20, -0.50, TABLE_HEIGHT + CUBE_HALF_SIZE[2]),
        half_size=CUBE_HALF_SIZE,
        color=(0.9, 0.1, 0.1, 1.0),
    ),
    "blue_block": make_block_cfg(
        name="blue_block",
        position=(0.10, -0.23, TABLE_HEIGHT + CUBE_HALF_SIZE[2]),
        half_size=CUBE_HALF_SIZE,
        color=(0.1, 0.3, 0.9, 1.0),
    ),
    "green_block": make_block_cfg(
        name="green_block",
        position=(-0.02, -0.35, TABLE_HEIGHT + CUBE_HALF_SIZE[2]),
        half_size=CUBE_HALF_SIZE,
        color=(0.1, 0.7, 0.2, 1.0),
    ),
    "yellow_block": make_block_cfg(
        name="yellow_block",
        position=(0.16, -0.38, TABLE_HEIGHT + CUBE_HALF_SIZE[2]),
        half_size=CUBE_HALF_SIZE,
        color=(0.95, 0.75, 0.05, 1.0),
    ),
    "small_red_block": make_block_cfg(
        name="small_red_block",
        position=(-0.18, -0.40, TABLE_HEIGHT + CUBE_HALF_SIZE[2]),
        half_size=CUBE_HALF_SIZE,
        color=(0.9, 0.1, 0.1, 1.0),
    ),
    "blue_cylinder": make_shape_cfg(
        spec_fn=lambda: get_cylinder_spec(
            name="blue_cylinder",
            radius=0.025,
            half_height=0.045,
            color=(0.08, 0.45, 0.95, 1.0),
        ),
        position=(-0.18, -0.29, TABLE_HEIGHT + 0.045),
    ),
    # New shapes stand upright: the 6 cm dimension is along world Z.
    "orange_prism": make_block_cfg(
        name="orange_prism",
        position=(0.26, -0.22, TABLE_HEIGHT + 0.03),
        half_size=(0.015, 0.015, 0.03),  # Full size: 3 x 3 x 6 cm.
        color=(0.95, 0.40, 0.05, 1.0),
    ),
    "purple_thin_prism": make_block_cfg(
        name="purple_thin_prism",
        position=(0.26, -0.34, TABLE_HEIGHT + 0.03),
        half_size=(0.0075, 0.015, 0.03),  # Full size: 1.5 x 3 x 6 cm.
        color=(0.65, 0.15, 0.80, 1.0),
    ),
    "cyan_cylinder": make_shape_cfg(
        spec_fn=lambda: get_cylinder_spec(
            name="cyan_cylinder",
            radius=0.015,
            half_height=0.03,
            color=(0.05, 0.75, 0.85, 1.0),
        ),
        position=(0.26, -0.48, TABLE_HEIGHT + 0.03),
    ),
    "yellow_tetrahedron": make_shape_cfg(
        spec_fn=lambda: get_tetrahedron_spec(
            name="yellow_tetrahedron",
            half_width=0.045,
            half_depth=0.040,
            height=0.065,
            color=(1.0, 0.82, 0.05, 1.0),
        ),
        position=(0.00, -0.17, TABLE_HEIGHT + 0.065 / 2),
    ),
    "red_arch_bridge": make_shape_cfg(
        spec_fn=lambda: get_arch_bridge_spec(
            name="red_arch_bridge",
            half_depth=0.022,
            color=(0.92, 0.12, 0.10, 1.0),
        ),
        # The shorter bridge extends 0.030 m below its body origin.
        position=(0.12, -0.48, TABLE_HEIGHT + 0.030),
    ),
    "green_hemisphere": make_shape_cfg(
        spec_fn=lambda: get_hemisphere_spec(
            name="green_hemisphere",
            radius=0.032,
            color=(0.1, 0.72, 0.3, 1.0),
        ),
        position=(-0.07, -0.48, TABLE_HEIGHT + 0.032 / 2),
    ),
}

def make_catalog_entities() -> dict[str, EntityCfg]:
    """One of each shape in each of the four colors; no extra movable objects."""
    result = {'robot': robot_cfg, 'table': table_cfg}
    for color_name, color in BLOCK_COLORS.items():
        for shape, dimensions in SHAPE_DIMENSIONS.items():
            name = f'{color_name}_{shape}'
            position = CATALOG_POSITIONS[name]
            if shape == 'cylinder':
                factory = lambda n=name, c=color, d=dimensions: get_cylinder_spec(n, d[0]/2, d[2]/2, c)
                result[name] = make_shape_cfg(factory, position)
            elif shape == 'triangular_prism':
                factory = lambda n=name, c=color, d=dimensions: get_triangular_prism_spec(n, d[0], c)
                result[name] = make_shape_cfg(factory, position)
            else:
                result[name] = make_block_cfg(name, position, tuple(d/2 for d in dimensions), color)
    return result


entities = make_catalog_entities()
scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=1.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities=entities,
)

def build_scene(tower: bool = False, prism: bool = False, *, legacy: bool = False,
                scene_task: str = 'random', object_count: int = 6, seed: int = 51,
                required_objects: tuple[str, ...] | None = None) -> tuple[Scene, mujoco.MjModel, mujoco.MjData]:
    """Compile and reset the shared SO-100 tabletop scene."""
    if sum((tower, prism, legacy)) > 1:
        raise ValueError('Choose only one historical task scene')
    from copy import deepcopy
    cfg = deepcopy(scene_cfg)
    if tower or prism or legacy:
        cfg.entities = deepcopy(legacy_entities)
    else:
        layout = generate_block_layout(seed, object_count, scene_task, required_objects=required_objects)
        cfg.entities = {name: cfg.entities[name] for name in ('robot', 'table', *layout.positions)}
        for name, position in layout.positions.items():
            cfg.entities[name].init_state.pos = position
    if prism:
        from aspire_so100.prism_layouts import PRISM_OBJECTS, PRISM_NOMINAL
        cfg.entities = {name: cfg.entities[name] for name in ('robot', 'table', *PRISM_OBJECTS)}
        for name, position in PRISM_NOMINAL.items():
            cfg.entities[name].init_state.pos = position
    if tower:
        # Historical three-cube construction workspace.
        cfg.entities = {name: cfg.entities[name] for name in ("robot", "table", "blue_block", "small_red_block")}
        cfg.entities["green_block"] = make_block_cfg(
            "green_block", (-0.10, -0.32, TABLE_HEIGHT + CUBE_HALF_SIZE[2]),
            CUBE_HALF_SIZE, (0.1, 0.7, 0.2, 1.0),
        )
    scene = Scene(cfg, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return scene, model, data


def main() -> None:
    import argparse
    from aspire_so100.tower_tasks import RECIPES, TowerRecipe, tower_recipe

    parser = argparse.ArgumentParser(description="Watch the SO-100 stacking task.")
    parser.add_argument("--scene-only", action="store_true", help="Open the passive scene without a policy.")
    parser.add_argument("--vision-task", choices=("tower", "prism", "inventory_prism", "inventory_tower"),
                        help="Sensor-only task (default: tower); inventory tasks use sampled scene goals")
    parser.add_argument('--scene-task', choices=('random', 'upright_thin_prism', 'catalog', 'single_column'),
                        help='New inventory scene: random subset, task-conditioned subset, or all 20 objects')
    parser.add_argument('--object-count', type=int, help='Subset size (default: 6); sampled passive scenes or inventory policies')
    parser.add_argument('--tower-recipe', choices=(*RECIPES, 'random'))
    parser.add_argument('--tower-order', nargs='+', help='Explicit inventory objects, bottom to top')
    parser.add_argument('--tower-levels', type=int, help='2..4 levels, only for the random tower recipe')
    parser.add_argument("--seed", type=int, help="Scene seed (default: 51, or 81 for --vision-policy).")
    parser.add_argument("--task", choices=("single", "tower"), default="single")
    parser.add_argument("--candidate", type=Path, help="Generated policy to replay (required for tower).")
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument("--vision-policy", type=Path,
                        help="Run a sensor-only RGB-D policy, then replay it in Viser.")
    vision.add_argument("--vision-trial", type=Path,
                        help="Replay an existing vision trial containing rollout.npz in Viser.")
    parser.add_argument("--vision-output", type=Path,
                        help="Keep the new vision trial in this non-existing directory.")
    parser.add_argument("--vision-layout", choices=("nominal", "random_xy", "sampled"),
                        help="Initial block positions when executing --vision-policy.")
    args = parser.parse_args()
    if args.scene_task is not None:
        if not args.scene_only or args.vision_task is not None:
            parser.error('--scene-task requires --scene-only without --vision-task')
        if args.scene_task == 'catalog' and args.object_count is not None:
            parser.error('The catalog always contains all 20 objects; omit --object-count')
    column_mode = args.vision_task == 'inventory_tower' or args.scene_task == 'single_column'
    if not column_mode and (args.tower_recipe is not None or args.tower_levels is not None or args.tower_order is not None):
        parser.error('Tower options require inventory_tower or --scene-task single_column')
    recipe = None
    if column_mode:
        try:
            if args.tower_order is not None:
                if args.tower_recipe is not None or args.tower_levels is not None:
                    raise ValueError('tower-order cannot be combined with recipe or levels')
                recipe = TowerRecipe('custom', tuple(args.tower_order))
            else:
                recipe = tower_recipe(args.tower_recipe or 'cube_triple',
                                      seed=args.seed if args.seed is not None else (81 if args.vision_policy else 51),
                                      levels=args.tower_levels)
            if (args.object_count or 6) < len(recipe.order):
                raise ValueError('object-count must include every recipe level')
        except ValueError as error:
            parser.error(str(error))
    inventory_policy = args.vision_policy is not None and args.vision_task in ('inventory_prism', 'inventory_tower')
    if args.vision_task in ('inventory_prism', 'inventory_tower') and not inventory_policy:
        parser.error('Inventory policy tasks require --vision-policy; use --scene-task for a passive scene')
    if args.object_count is not None:
        if not ((args.scene_only and args.vision_task is None) or inventory_policy):
            parser.error('--object-count requires a sampled passive scene or inventory policy')
        if inventory_policy and not 2 <= args.object_count <= 20:
            parser.error('Inventory tasks require 2..20 objects')
    if args.scene_only and args.vision_task is None:
        try:
            generate_block_layout(args.seed if args.seed is not None else 51,
                                  args.object_count if args.object_count is not None else 6,
                                  args.scene_task or 'random', required_objects=None if recipe is None else recipe.order)
        except ValueError as error:
            parser.error(str(error))
    if args.vision_output is not None and args.vision_policy is None:
        parser.error("--vision-output requires --vision-policy")
    if (args.vision_policy or args.vision_trial) and (args.scene_only or args.candidate or args.task != "single"):
        parser.error("Vision modes cannot be combined with --scene-only, --candidate, or --task")
    if args.vision_policy:
        layout = args.vision_layout or ('sampled' if inventory_policy else 'nominal')
        if inventory_policy != (layout == 'sampled'):
            parser.error('Only inventory tasks use --vision-layout sampled')
        from aspire_so100.playback import watch_vision_policy
        watch_vision_policy(args.vision_policy, seed=args.seed if args.seed is not None else 81,
                            output=args.vision_output, layout=layout, task=args.vision_task or 'tower',
                            object_count=args.object_count, tower_recipe=args.tower_recipe, tower_levels=args.tower_levels,
                            tower_order=args.tower_order)
        return
    if args.vision_trial:
        from aspire_so100.playback import watch_vision_rollout
        watch_vision_rollout(args.vision_trial)
        return
    if not args.scene_only:
        from aspire_so100.playback import watch_stacking

        if args.task == "tower" and args.candidate is None:
            parser.error("--task tower requires --candidate pointing to an LLM-generated policy")
        watch_stacking(args.seed if args.seed is not None else 51,
                       task=args.task, candidate_path=args.candidate)
        return
    _, model, data = build_scene(tower=args.vision_task == 'tower', prism=args.vision_task == 'prism',
                               scene_task=args.scene_task or 'random',
                               object_count=args.object_count if args.object_count is not None else 6,
                               seed=args.seed if args.seed is not None else 51,
                               required_objects=None if recipe is None else recipe.order)
    print(
        f"Scene compiled: "
        f"{model.nbody} bodies, "
        f"{model.ngeom} geometries, "
        f"{model.nu} actuators"
    )
    from aspire_so100.camera_view import run_camera_viewer

    run_camera_viewer(model, data)


if __name__ == "__main__":
    main()
