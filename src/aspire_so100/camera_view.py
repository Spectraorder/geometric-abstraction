"""Wrist RGB and fixed overhead RGB-D views for Viser."""

import time

import mujoco
import numpy as np
import viser
from mjviser import Viewer
from aspire_so100.rgbd import OverheadRGBD, depth_preview


class WristCameraView:
    def __init__(self, server, model, data):
        self.model, self.data = model, data
        self.camera_id = model.camera("robot/wrist_camera").id
        self.renderer = mujoco.Renderer(model, height=240, width=320)
        self.last_frame = -float("inf")
        self.frustum = server.scene.add_camera_frustum(
            "/wrist_camera",
            fov=np.deg2rad(model.cam_fovy[self.camera_id]),
            aspect=320 / 240,
            scale=0.07,
            thickness=0.001,
            color=(255, 180, 30),
        )
        with server.gui.add_folder("Arm camera"):
            server.gui.add_markdown("Live wrist view. The orange outline shows its position and field of view.")
            self.enabled = server.gui.add_checkbox("Show camera image", initial_value=True)
            self.image = server.gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label="Wrist camera")

    def update(self, scene):
        scene.update_from_mjdata(self.data)
        # MuJoCo: -Z forward, +Y up. Viser: +Z forward, +Y down.
        rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        position = self.data.cam_xpos[self.camera_id].copy()
        # Match mjviser's translation when a body is being tracked.
        tracked = scene._tracked_body_id
        if scene.camera_tracking_enabled and tracked is not None:
            position -= self.data.xpos[tracked]
        self.frustum.position = position
        self.frustum.wxyz = quaternion
        self.image.visible = self.enabled.value
        now = time.monotonic()
        if self.enabled.value and now - self.last_frame >= 0.1:
            self.renderer.update_scene(self.data, camera=self.camera_id)
            self.image.image = self.renderer.render().copy()
            self.last_frame = now

    def close(self):
        self.renderer.close()


class OverheadCameraView:
    def __init__(self, server, model, data):
        self.data = data
        self.sensor = OverheadRGBD(model)
        self.last_frame = -float("inf")
        self.frustum = server.scene.add_camera_frustum(
            "/overhead_camera", fov=np.deg2rad(model.cam_fovy[self.sensor.camera_id]),
            aspect=320 / 240, scale=.12, thickness=.002, color=(40, 210, 255),
        )
        with server.gui.add_folder("Side-mounted RGB-D camera"):
            server.gui.add_markdown(
                "Raised side view on the right-hand mast, aimed at the workspace. Cyan outline: camera pose. "
                "Ideal simulated depth along the optical axis; no sensor noise. "
                "Depth display: white ≤0.25 m, black ≥1.00 m. Table depth varies across the image."
            )
            self.enabled = server.gui.add_checkbox("Show side-camera images", initial_value=True)
            self.rgb = server.gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label="Side camera RGB")
            self.depth = server.gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label="Side camera depth (m)")

    def update(self, scene):
        self.rgb.visible = self.depth.visible = self.enabled.value
        # Follow the viewer's tracking translation even while image updates are off.
        camera_id = self.sensor.camera_id
        position = self.data.cam_xpos[camera_id].copy()
        rotation = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        tracked = scene._tracked_body_id
        if scene.camera_tracking_enabled and tracked is not None:
            position -= self.data.xpos[tracked]
        self.frustum.position, self.frustum.wxyz = position, quat
        now = time.monotonic()
        if self.enabled.value and now - self.last_frame >= .2:
            frame = self.sensor.capture(self.data)
            self.rgb.image = frame["rgb"]
            self.depth.image = depth_preview(frame["depth_m"])
            self.last_frame = now

    def close(self):
        self.sensor.close()


def run_camera_viewer(model, data, **kwargs):
    mujoco.mj_forward(model, data)
    server = viser.ViserServer()
    camera = None
    overhead = None
    try:
        camera = WristCameraView(server, model, data)
        overhead = OverheadCameraView(server, model, data)
        def update(scene):
            camera.update(scene)
            overhead.update(scene)
        viewer = Viewer(model, data, server=server, render_fn=update, **kwargs)
        viewer.run()
    finally:
        if camera is not None:
            camera.close()
        if overhead is not None:
            overhead.close()
        server.stop()
