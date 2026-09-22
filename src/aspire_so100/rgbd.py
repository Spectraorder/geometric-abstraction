"""Aligned ideal-simulator RGB and metric optical-axis depth."""
import mujoco
import numpy as np


class OverheadRGBD:
    def __init__(self, model, width=320, height=240):
        self.model = model
        self.camera_id = model.camera("table/overhead_camera").id
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture(self, data):
        mujoco.mj_forward(self.model, data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        try:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=self.camera_id)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        focal = .5 * self.height / np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id]) / 2)
        K = np.array([[focal, 0., (self.width - 1) / 2],
                      [0., focal, (self.height - 1) / 2], [0., 0., 1.]])
        # Pixel coordinates: right, down; camera coordinates: right, down, forward.
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        pose[:3, 3] = data.cam_xpos[self.camera_id]
        far = self.model.vis.map.zfar * self.model.stat.extent
        return {"rgb": rgb, "depth_m": depth,
                "valid_depth": np.isfinite(depth) & (depth > 0) & (depth < far * .999),
                "intrinsics": K, "camera_to_world": pose, "time": float(data.time)}

    def close(self):
        self.renderer.close()

    def capture_wrist(self, data):
        """RGB and robot-camera calibration only; the wrist has no depth sensor."""
        camera_id = self.model.camera("robot/wrist_camera").id
        mujoco.mj_forward(self.model, data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=camera_id)
        focal = .5 * self.height / np.tan(np.deg2rad(self.model.cam_fovy[camera_id]) / 2)
        pose = np.eye(4)
        pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        pose[:3, 3] = data.cam_xpos[camera_id]
        return {"rgb": self.renderer.render().copy(),
                "intrinsics": np.array([[focal, 0., (self.width - 1) / 2],
                                        [0., focal, (self.height - 1) / 2], [0., 0., 1.]]),
                "camera_to_world": pose, "time": float(data.time)}


def depth_preview(depth, near=.25, far=1.0):
    """Fixed display scale only: near white, far black. Raw depth stays metric."""
    scaled = 1. - (np.nan_to_num(depth, nan=far, posinf=far, neginf=far) - near) / (far - near)
    pixels = np.uint8(np.clip(scaled, 0., 1.) * 255)
    return np.repeat(pixels[..., None], 3, axis=-1)


def save_visual_observation(env, path, label):
    """One image sheet with side RGB, depth preview and wrist RGB; raw depth alongside."""
    from PIL import Image, ImageDraw
    frame = env.get_overhead_rgbd()
    renderer = env._overhead_rgbd.renderer
    renderer.disable_depth_rendering()
    renderer.update_scene(env.data, camera="robot/wrist_camera")
    wrist = renderer.render().copy()
    sheet = Image.new("RGB", (960, 272), "white")
    draw = ImageDraw.Draw(sheet)
    for x, title, pixels in [(0,"Side RGB",frame["rgb"]),
                              (320,"Depth: white 0.25m / black 1.0m",depth_preview(frame["depth_m"])),
                              (640,"Wrist RGB",wrist)]:
        draw.text((x+4,3), title, fill="black")
        sheet.paste(Image.fromarray(pixels), (x,32))
    draw.text((4,17), f"{label}; seed {env.seed}; t={frame['time']:.2f}s", fill="black")
    sheet.save(path)
    np.savez_compressed(path.with_suffix('.npz'), depth_m=frame['depth_m'],
                        valid_depth=frame['valid_depth'], intrinsics=frame['intrinsics'],
                        camera_to_world=frame['camera_to_world'])
    return str(path.resolve())
