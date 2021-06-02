import os
import sys
import numpy as np
from typing import Any, Dict, List, Tuple, Union
import cv2

import gym
from gym import error, spaces

from mlagents_envs.base_env import BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .gym_unity import UnityToGymWrapper, GymStepResult


class DroneDodgeBall(UnityToGymWrapper):
    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        stacked_vec_obs: int = 1,
        use_ball_pose: bool = False, # use ground-truth ball pose; you can use no_graphics
        use_held_ball: bool = True, # use smaller held ball
        input_mode: int = 3, # 1: thrust only, 2: thrust and roll, 3: thrust and all axes
        no_graphics: bool = False,
        log_folder: str = None
    ):
        dirname = os.path.dirname(__file__)
        if sys.platform == "darwin":
            env_build = os.path.join(dirname, '../../builds/DroneDodgeBall/Mac')
        elif sys.platform == 'win32':
            env_build = os.path.join(dirname, '..\\..\\builds\\DroneDodgeBall\\Windows\\DroneDodgeBall.exe')
        elif sys.platform == 'linux':
            env_build =  os.path.join(dirname, '../../builds/DroneDodgeBall/Linux/DroneDodgeBall.x86_64')
        channel = EnvironmentParametersChannel()
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[channel],
                                     log_folder=log_folder)
        channel_inp = 1.0 if use_ball_pose else 0.0
        channel.set_float_parameter("use_ball_pose", channel_inp)
        channel_inp = 1.0 if use_held_ball else 0.0
        channel.set_float_parameter("use_held_ball", channel_inp)
        channel.set_float_parameter("input_mode", float(input_mode))
        super(DroneDodgeBall, self).__init__(unity_env=unity_env,
                                             uint8_visual=uint8_visual,
                                             flatten_branched=flatten_branched,
                                             allow_multiple_obs=allow_multiple_obs,
                                             stacked_vec_obs=stacked_vec_obs)

        ball_pose_idx = -6
        if use_ball_pose: # drop raycast observation
            low = self._observation_space.low[ball_pose_idx:]
            high = self._observation_space.high[ball_pose_idx:]
            dtype = self._observation_space.dtype
            self._observation_space = spaces.Box(low, high, dtype=dtype)
        else: # drop ball pose
            low = self._observation_space.low[:ball_pose_idx]
            high = self._observation_space.high[:ball_pose_idx]
            dtype = self._observation_space.dtype
            self._observation_space = spaces.Box(low, high, dtype=dtype)
        self._use_ball_pose = use_ball_pose
        self._ball_pose_idx = ball_pose_idx

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)
        if self._use_ball_pose:
            obs = obs[self._ball_pose_idx:]
        else:
            obs = obs[:self._ball_pose_idx]

        return (obs, rew, done, _info)
