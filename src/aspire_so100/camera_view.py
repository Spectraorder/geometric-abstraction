"""Wrist RGB and fixed overhead RGB-D views for Viser."""

import html
import time

import mujoco
import numpy as np
import viser
from mjviser import Viewer
from mjviser.conversions import merge_geoms
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


class TrajectoryReviewView:
    """Offline plan-versus-execution overlay for a saved policy rollout."""
    def __init__(self, server, model, data, review, seek_fn, playback_index_fn):
        self.server = server
        self.model, self.data = model, data
        self.review = review
        self.seek_fn = seek_fn
        self.playback_index_fn = playback_index_fn
        positions=np.asarray(review['positions'],float)
        times=np.asarray(review['times'],float)
        self.root=server.scene.add_frame('/trajectory',show_axes=False)
        # Decimate only the drawing; the green cursor still uses every replay frame.
        draw=np.unique(np.r_[np.arange(0,len(positions),max(1,len(positions)//600)),len(positions)-1])
        path=positions[draw]
        progress=(times[draw]/max(review['duration'],1e-9))[:,None]
        colors=np.clip(np.c_[80+150*progress[:,0], 80+40*(1-progress[:,0]),
                             255-60*progress[:,0]],0,255).astype(np.uint8)
        segments=np.stack((path[:-1],path[1:]),axis=1) if len(path)>1 else np.empty((0,2,3))
        segment_colors=np.stack((colors[:-1],colors[1:]),axis=1) if len(path)>1 else np.empty((0,2,3),np.uint8)
        self._draw_indices=draw
        self._draw_path=path
        self._draw_colors=colors
        self._executed_segments=segments
        self._executed_colors=segment_colors
        self._executed_end_times=times[draw][1:]
        self._executed_count=-1
        self._tail_visible=False
        self.executed=server.scene.add_line_segments('/trajectory/executed',segments[:0],
            segment_colors[:0],thickness=3.,thickness_units='screen',visible=False)
        tail_points=np.asarray([[positions[0],positions[0]]],float)
        tail_colors=np.asarray([[colors[0],colors[0]]],np.uint8)
        self.executed_tail=server.scene.add_line_segments('/trajectory/executed_tail',
            tail_points,tail_colors,thickness=3.,thickness_units='screen',visible=False)
        motion_segments=[]
        for motion in review['motions']:
            index=int(np.argmin(abs(times-motion['time'])))
            motion_segments.append([positions[index],motion['position']])
        has_motions=bool(motion_segments)
        self._motion_points=(np.asarray(motion_segments,float).reshape(-1,2,3) if has_motions
                            else np.empty((0,2,3),float))
        self._motion_times=np.asarray([motion['time'] for motion in review['motions']],float)
        self._commanded_count=-1
        self.commanded=server.scene.add_line_segments('/trajectory/commanded',
            self._motion_points[:0], (255,160,40), thickness=4.,thickness_units='screen',
            visible=False)
        self._phantom_handles=[]
        self._phantom_index=-1
        self._phantom_visible=False
        robot_body_ids=np.asarray(review.get('robot_body_ids',[]),int)
        for body_column,body_id in enumerate(robot_body_ids):
            geom_ids=np.flatnonzero(model.geom_bodyid==body_id)
            if not len(geom_ids):
                continue
            groups=model.geom_group[geom_ids]
            visual_group=2 if np.any(groups==2) else int(np.min(groups))
            geom_ids=geom_ids[groups==visual_group].tolist()
            mesh=merge_geoms(model,geom_ids)
            body_name=(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(body_id))
                       or f'body_{body_id}')
            handle=server.scene.add_mesh_simple(
                f'/trajectory/phantom/{body_name}',np.asarray(mesh.vertices),
                np.asarray(mesh.faces),color=(80,210,255),opacity=.24,
                material='toon3',side='double',cast_shadow=False,receive_shadow=False,
                position=review['robot_positions'][0,body_column],
                wxyz=review['robot_quaternions'][0,body_column],visible=False)
            self._phantom_handles.append((handle,body_column))
        self.cursor=server.scene.add_icosphere('/trajectory/current',radius=.009,color=(70,240,140),
            position=positions[0])
        with server.gui.add_folder('Trajectory review',expand_by_default=True):
            server.gui.add_markdown(
                '**Purple:** executed TCP path · **Orange:** commanded motions · '
                '**Green:** current replay position.  '
                'Pause in Controls before scrubbing. Times are simulated seconds from policy start.')
            self.executed_toggle=server.gui.add_checkbox('Executed path',initial_value=True)
            self.commanded_toggle=server.gui.add_checkbox('Commanded motion',initial_value=has_motions)
            self.phantom_toggle=server.gui.add_checkbox('Look-ahead phantom',
                                                        initial_value=bool(self._phantom_handles))
            self.phantom_lookahead=server.gui.add_slider('Phantom look-ahead (s)',
                min=.1,max=3.,step=.1,initial_value=1.)
            policy_end=min(max(float(review['policy_end']),0.),float(review['duration']))
            marks=((round(policy_end,2),'policy end'),) if review['duration'] else None
            self.slider=server.gui.add_slider('Review time (s)',min=0.,max=max(.01,review['duration']),
                step=.01,initial_value=0.,marks=marks)
            self.previous=server.gui.add_button('Previous policy event')
            self.next=server.gui.add_button('Next policy event')
            self.status=server.gui.add_html('')
        self._updating_slider=False

        @self.executed_toggle.on_update
        def _(_):
            enabled=self.executed_toggle.value
            self.executed.visible=enabled and self._executed_count>0
            self.executed_tail.visible=enabled and self._tail_visible
        @self.commanded_toggle.on_update
        def _(_): self.commanded.visible=(self.commanded_toggle.value and
                                          self._commanded_count>0)
        @self.phantom_toggle.on_update
        def _(_):
            for handle,_ in self._phantom_handles:
                handle.visible=self.phantom_toggle.value and self._phantom_visible
        @self.phantom_lookahead.on_update
        def _(_):
            index=int(np.clip(self.playback_index_fn(),0,len(self.review['times'])-1))
            self._phantom_index=-1
            self._update_phantom(index,float(self.review['times'][index]))
        @self.slider.on_update
        def _(_):
            if not self._updating_slider:
                self.seek_fn(self.model,self.data,float(self.slider.value))
        @self.previous.on_click
        def _(_): self._seek_event(-1)
        @self.next.on_click
        def _(_): self._seek_event(1)

    def _seek_event(self,direction):
        now=float(self.slider.value)
        event_times=sorted(set(float(e['time']) for e in self.review['events']))
        candidates=([t for t in event_times if t<now-.005] if direction<0
                    else [t for t in event_times if t>now+.005])
        target=(candidates[-1] if direction<0 and candidates else
                candidates[0] if direction>0 and candidates else now)
        self.slider.value=float(np.clip(target,0,self.review['duration']))
        self.seek_fn(self.model,self.data,self.slider.value)

    def _update_phantom(self,index,now):
        future_time=min(now+float(self.phantom_lookahead.value),self.review['duration'])
        future_index=int(np.searchsorted(self.review['times'],future_time,side='left'))
        future_index=min(future_index,len(self.review['times'])-1)
        self._phantom_visible=(future_index>index and
                               now<self.review['duration']-1e-9 and
                               bool(self._phantom_handles))
        if future_index!=self._phantom_index:
            self._phantom_index=future_index
            with self.server.atomic():
                for handle,body_column in self._phantom_handles:
                    handle.position=self.review['robot_positions'][future_index,body_column]
                    handle.wxyz=self.review['robot_quaternions'][future_index,body_column]
        visible=self.phantom_toggle.value and self._phantom_visible
        for handle,_ in self._phantom_handles:
            handle.visible=visible

    def update(self,scene):
        index=int(np.clip(self.playback_index_fn(),0,len(self.review['times'])-1))
        now=float(self.review['times'][index])
        point=self.review['positions'][index]
        self.cursor.position=point
        executed_count=int(np.searchsorted(self._executed_end_times,now,side='right'))
        if executed_count!=self._executed_count:
            self._executed_count=executed_count
            with self.server.atomic():
                self.executed.points=self._executed_segments[:executed_count]
                self.executed.colors=self._executed_colors[:executed_count]
        self.executed.visible=self.executed_toggle.value and executed_count>0
        tail_start_index=int(self._draw_indices[min(executed_count,len(self._draw_indices)-1)])
        self._tail_visible=(index>tail_start_index and executed_count<len(self._draw_path)-1)
        if self._tail_visible:
            tail_progress=now/max(float(self.review['duration']),1e-9)
            tail_color=np.clip([80+150*tail_progress,80+40*(1-tail_progress),
                                255-60*tail_progress],0,255).astype(np.uint8)
            self.executed_tail.points=np.asarray([[self._draw_path[executed_count],point]])
            self.executed_tail.colors=np.asarray([[self._draw_colors[executed_count],tail_color]])
        self.executed_tail.visible=self.executed_toggle.value and self._tail_visible
        commanded_count=int(np.searchsorted(self._motion_times,now,side='right'))
        if commanded_count!=self._commanded_count:
            self._commanded_count=commanded_count
            self.commanded.points=self._motion_points[:commanded_count]
        self.commanded.visible=self.commanded_toggle.value and commanded_count>0
        self._update_phantom(index,now)
        self._updating_slider=True
        self.slider.value=round(now,2)
        self._updating_slider=False
        offset=np.zeros(3)
        tracked=scene._tracked_body_id
        if scene.camera_tracking_enabled and tracked is not None:
            offset=-self.data.xpos[tracked]
        self.root.position=offset
        events=self.review['events']
        previous=max((e for e in events if e['time']<=now),key=lambda e:e['time'],default=None)
        upcoming=min((e for e in events if e['time']>now),key=lambda e:e['time'],default=None)
        phase=('policy execution' if now<=self.review['policy_end']+.005
               else 'post-policy evaluator hold')
        prev_text='none' if previous is None else f"{previous['time']:.2f}s {previous['label']}"
        next_text='none' if upcoming is None else f"{upcoming['time']:.2f}s {upcoming['label']}"
        self.status.content=(f'<b>{now:.2f} / {self.review["duration"]:.2f} s</b> · {phase}<br>'
                             f'<b>Last:</b> {html.escape(prev_text)}<br>'
                             f'<b>Next:</b> {html.escape(next_text)}')


def run_camera_viewer(model, data, trajectory_review=None, seek_fn=None,
                      playback_index_fn=None, **kwargs):
    mujoco.mj_forward(model, data)
    server = viser.ViserServer()
    camera = None
    overhead = None
    trajectory = None
    try:
        camera = WristCameraView(server, model, data)
        overhead = OverheadCameraView(server, model, data)
        if trajectory_review is not None:
            if seek_fn is None or playback_index_fn is None:
                raise ValueError('Trajectory review requires seek and playback-index callbacks')
            trajectory=TrajectoryReviewView(server,model,data,trajectory_review,
                                            seek_fn,playback_index_fn)
        def update(scene):
            camera.update(scene)
            overhead.update(scene)
            if trajectory is not None:
                trajectory.update(scene)
        viewer = Viewer(model, data, server=server, render_fn=update, **kwargs)
        viewer.run()
    finally:
        if camera is not None:
            camera.close()
        if overhead is not None:
            overhead.close()
        server.stop()
