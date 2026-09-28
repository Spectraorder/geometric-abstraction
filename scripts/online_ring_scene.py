"""Continuous Ring episode with one reset and a private stability monitor."""
import math
import time
import numpy as np
from ring_peg_scene import RingPegEnv
from online_bridge_scene import EpisodeSuccess
from online_contact_scene import ContactCompleteEnv


class OnlineRingEnv(RingPegEnv):
    def __init__(self, output_dir, *, scene_seed):
        self.reset_count = 0
        self.episode_started = self.deadline = None
        self.realtime = True
        self.total_steps = 0
        self.max_simulation_lag_s = 0.
        self.stable_since = self.stable_position = None
        self.success_time = self.success_evidence = None
        self.last_success_check = -math.inf
        self.monitor_rows = []
        super().__init__(output_dir, scene_seed=scene_seed)

    def reset(self, seed=None):
        if self.reset_count:
            raise RuntimeError('Online episodes permit exactly one initial reset')
        self.reset_count += 1
        return super().reset(seed)

    def start_episode(self, budget_s=200., *, realtime=True):
        if self.reset_count != 1 or self.episode_started is not None:
            raise RuntimeError('Start once after the initial reset')
        if not math.isfinite(budget_s) or not 0 < budget_s <= 200:
            raise ValueError('Budget must be positive and at most 200 seconds')
        self.episode_started = time.monotonic()
        self.deadline = self.episode_started + budget_s
        self.initial_sim_time = float(self.data.time)
        self.initial_step_count = self.total_steps
        self.realtime = realtime

    def _step(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('Online episode budget exhausted')
        super()._step()
        self.total_steps += 1
        if self.episode_started is None:
            return
        target = self.episode_started + float(self.data.time) - self.initial_sim_time
        now = time.monotonic()
        self.max_simulation_lag_s = max(self.max_simulation_lag_s, now-target)
        if self.realtime and target > now:
            time.sleep(min(target-now, max(0., self.deadline-now)))
        self._monitor_success()
        if self.success_time is not None:
            raise EpisodeSuccess('Ring remained released over the peg for two seconds')

    def _monitor_success(self):
        now = float(self.data.time)
        if now-self.last_success_check < .1:
            return
        self.last_success_check = now
        status = self.ring_status()
        position = np.asarray(status['ring_center_m'])
        stable = status['on_peg']
        drift = 0. if self.stable_position is None else float(np.linalg.norm(position-self.stable_position))
        self.monitor_rows.append(dict(time=now, wall_seconds=time.monotonic()-self.episode_started,
                                      on_peg=stable, drift_m=drift, reasons=status['reasons']))
        if not stable or drift > .003:
            self.stable_since = self.stable_position = None
        elif self.stable_since is None:
            self.stable_since, self.stable_position = now, position.copy()
        elif now-self.stable_since >= 2.:
            self.success_time = time.monotonic()-self.episode_started
            self.success_evidence = dict(start_sim_time=self.stable_since, end_sim_time=now,
                                         drift_m=drift, hold_s=2., drift_threshold_m=.003)

    def advance_while_waiting(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Online episode budget exhausted')
        if self.realtime:
            target = self.initial_sim_time + time.monotonic()-self.episode_started
            while float(self.data.time) < target:
                self._step()
            time.sleep(min(.005, max(0., self.deadline-time.monotonic())))
        else:
            self._step()

    def timing(self):
        return dict(reset_count=self.reset_count, physics_steps=self.total_steps-self.initial_step_count,
                    wall_seconds=time.monotonic()-self.episode_started,
                    simulated_seconds=float(self.data.time)-self.initial_sim_time,
                    max_simulation_lag_seconds=self.max_simulation_lag_s,
                    physics_during_model_wait=True, paced_actions=self.realtime)

    archive_terminal_physics = ContactCompleteEnv.archive_terminal_physics
