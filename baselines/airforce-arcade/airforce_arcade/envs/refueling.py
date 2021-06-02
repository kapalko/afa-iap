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


class Refueling(UnityToGymWrapper):
    DISTANCE_MAX = 1000.

    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        stacked_vec_obs: int = 1,
        no_graphics: bool = False,
        log_folder: str = None,
        input_mode: int = 3, # (1) pitch-only control (2) pitch and throttle (3) all controls
        randomize_fighter_start_pose: bool = True,
        randomize_fighter_start_thrust: bool = False,
        reward_mode: int = 1
    ):
        dirname = os.path.dirname(__file__)
        if sys.platform == "darwin":
            env_build = os.path.join(dirname, '../../builds/Refueling/Mac')
        elif sys.platform == 'win32':
            env_build = os.path.join(dirname, '..\\..\\builds\\Refueling\\Windows\\Refueling.exe')
        elif sys.platform == 'linux':
            env_build =  os.path.join(dirname, '../../builds/Refueling/Linux/Refueling.x86_64')
        channel = EnvironmentParametersChannel()
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[channel],
                                     log_folder=log_folder)
        channel.set_float_parameter("input_mode", float(input_mode))
        channel_inp = 1. if randomize_fighter_start_pose else 0.
        channel.set_float_parameter("randomize_fighter_start_pose", channel_inp)
        channel_inp = 1. if randomize_fighter_start_thrust else 0.
        channel.set_float_parameter("randomize_fighter_start_thrust", channel_inp)
        super(Refueling, self).__init__(unity_env=unity_env,
                                        uint8_visual=uint8_visual,
                                        flatten_branched=flatten_branched,
                                        allow_multiple_obs=allow_multiple_obs,
                                        stacked_vec_obs=stacked_vec_obs)
        
        if input_mode == 1:
            # don't change self.action_size as this is used internally in the class
            high = np.array([1] * 1)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        elif input_mode == 2:
            high = np.array([1] * 2)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        self._input_mode = input_mode
        self._reward_mode = reward_mode

    def step(self, action: List[Any]) -> GymStepResult:
        dummy = 0.
        if self._input_mode == 1:
            action = np.array([action[0], dummy, dummy]) # pitch
        elif self._input_mode == 2:
            action = np.array([action[0], action[1], dummy]) # pitch + throttle

        return super().step(action)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)

        self._recompute_reward(obs, rew)

        return (obs, rew, done, _info)

    def _recompute_reward(self, obs, rew):
        distance_to_edge = obs[-1] * self.DISTANCE_MAX # NOTE: remember to accomodate stacked observation
        distance_thresh = 10.
        in_envelope = distance_to_edge <= distance_thresh
        if self._reward_mode == 1: # original reward (inverse distance reward + survival reward 0.01)
            rew = rew
        elif self._reward_mode == 2: # inverse distance reward
            distance_rew = 1. / max(1., distance_to_edge)
            rew = distance_rew
        elif self._reward_mode == 3: # sparse reward
            rew = 1. if in_envelope else 0.0
        elif self._reward_mode == 4: # linear distance reward
            multiplier = 0.1
            clipped_distance = max(distance_thresh, distance_to_edge)
            rew = (self.DISTANCE_MAX - clipped_distance) / self.DISTANCE_MAX * multiplier
        elif self._reward_mode == 5: # incremental distance reward
            if not hasattr(self, "prev_distance_to_edge"):
                self.prev_distance_to_edge = distance_to_edge
            delta_distance = self.prev_distance_to_edge - distance_to_edge
            rew = (delta_distance / self.DISTANCE_MAX) if distance_to_edge > distance_thresh else 1.
            self.prev_distance_to_edge = distance_to_edge
        else:
            raise NotImplementedError("Reward mode {} is not supported".format(self._reward_mode))

        return rew