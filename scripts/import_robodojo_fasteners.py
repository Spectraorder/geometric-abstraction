"""Convert pinned RoboDojo USDZ geometry; run in an isolated usd-core environment.

No decimation, deformation or procedural thread replacement. Only USD transforms,
unit conversion, triangulation and a common bottom-center origin are applied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
REVISION = '43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab'


def extract(stage, prim):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    vertices = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
    vertices *= UsdGeom.GetStageMetersPerUnit(stage)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    normals = np.asarray(mesh.GetNormalsAttr().Get() or [], dtype=float).reshape(-1, 3)
    interpolation = mesh.GetNormalsInterpolation()
    faces, normal_faces = [], []
    offset = 0
    for face_index, count in enumerate(counts):
        for i in range(1, count-1):
            corners = np.array([offset, offset+i, offset+i+1])
            faces.append(indices[corners])
            if len(normals):
                if interpolation == 'faceVarying': normal_faces.append(corners)
                elif interpolation == 'vertex': normal_faces.append(indices[corners])
                elif interpolation == 'uniform': normal_faces.append([face_index]*3)
                elif interpolation == 'constant': normal_faces.append([0]*3)
                else: raise ValueError(f'Unsupported normals: {interpolation}')
        offset += count
    faces = np.asarray(faces, dtype=np.int32)
    normal_faces = np.asarray(normal_faces, dtype=np.int32).reshape(-1, 3)
    if (mesh.GetOrientationAttr().Get() == 'leftHanded') != (np.linalg.det(transform[:3,:3]) < 0):
        faces = faces[:, ::-1]
        normal_faces = normal_faces[:, ::-1]
    if len(normals):
        normals = normals @ np.linalg.inv(transform[:3,:3]).T
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1.e-15)
        # Some upstream corners have zero authored normals. Repair only their
        # shading directions; keep every source vertex and triangle unchanged.
        triangle_normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                                    vertices[faces[:,2]]-vertices[faces[:,0]])
        triangle_lengths = np.linalg.norm(triangle_normals, axis=1, keepdims=True)
        triangle_normals /= np.maximum(triangle_lengths, 1.e-15)
        triangle_normals[triangle_lengths[:,0] < 1.e-15] = (0.,0.,1.)
        bad = lengths[normal_faces,0] < 1.e-15
        normals[normal_faces[bad]] = np.broadcast_to(triangle_normals[:,None,:],
                                                    (*normal_faces.shape,3))[bad]
    return dict(vertices=vertices, faces=faces, normals=normals, normal_faces=normal_faces)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=ROOT/'assets/robodojo_fasteners/upstream')
    p.add_argument('--output', type=Path, default=ROOT/'assets/robodojo_fasteners/converted')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(repository='RoboDojo-Benchmark/RoboDojo', revision=REVISION,
                    dataset_license='Apache-2.0', transform='USD world transform; meters; bottom-center origin',
                    parts={})
    for kind, name in (('Rigid','factory_nut'), ('Geometry','factory_bolt')):
        relative = Path('Assets/Object/RoboDojo')/kind/name/'00000/object.usdz'
        source = a.upstream/relative
        stage = Usd.Stage.Open(str(source))
        if UsdGeom.GetStageUpAxis(stage) != 'Z':
            raise ValueError('Expected Z-up source assets')
        visual_prims = [prim for prim in stage.Traverse()
                        if prim.IsA(UsdGeom.Mesh) and '/visual/' in str(prim.GetPath())]
        if len(visual_prims) != 1:
            raise ValueError('Expected one original visible mesh per fastener')
        mesh = extract(stage, visual_prims[0])
        low, high = mesh['vertices'].min(0), mesh['vertices'].max(0)
        shift = -np.r_[(low[:2]+high[:2])/2, low[2]]
        mesh['vertices'] += shift
        destination = a.output/f'{name}.npz'
        np.savez_compressed(destination, **mesh)
        proxies = []
        # Preserve RoboDojo's six authored convex edge boxes for nut grasping.
        # Its concave SDF collider cannot be used as a single MuJoCo convex mesh.
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and '/collision/edge_cube_' in str(prim.GetPath()):
                proxy = extract(stage, prim)
                proxy['vertices'] += shift
                filename = f'{name}_{prim.GetName()}.npz'
                np.savez_compressed(a.output/filename, **proxy)
                proxies.append(filename)
        manifest['parts'][name] = dict(
            source_path=str(relative), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            mesh_file=destination.name, mesh_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            visual_prim=str(visual_prims[0].GetPath()), origin_shift_m=shift.tolist(),
            bounds_m=[mesh['vertices'].min(0).tolist(),mesh['vertices'].max(0).tolist()],
            vertices=len(mesh['vertices']), triangles=len(mesh['faces']),
            grasp_collision_proxies=proxies)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
