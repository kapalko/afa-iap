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


class CanyonRun(UnityToGymWrapper):
    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        no_graphics: bool = False,
        log_folder: str = None,
        input_mode: int = 2, # (1) pitch-control only environment/straight mode (2) pitch, roll, (3) pitch, roll, yaw, and throttle
        survival_reward: float = 0.002, # reward given every frame update for surviving
        waypoint_reward: float =  0.5 # reward given for reaching a waypoint
    ):
        dirname = os.path.dirname(__file__)
        if sys.platform == "darwin":
            env_build = os.path.join(dirname, '../../builds/CanyonRun/Mac')
        elif sys.platform == 'win32':
            env_build = os.path.join(dirname, '..\\..\\builds\\InfiniteMazeScene\\Windows\\CanyonRun.exe')
        elif sys.platform == 'linux':
            env_build =  os.path.join(dirname, '../../builds/InfiniteMazeScene/Linux/CanyonRun.x86_64')
        channel = EnvironmentParametersChannel()
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[channel],
                                     log_folder=log_folder)
        channel.set_float_parameter("input_mode", float(input_mode))
        channel.set_float_parameter("survival_reward", survival_reward)
        channel.set_float_parameter("waypoint_reward", waypoint_reward)

        super(CanyonRun, self).__init__(unity_env=unity_env,
                                                uint8_visual=uint8_visual,
                                                flatten_branched=flatten_branched,
                                                allow_multiple_obs=allow_multiple_obs)


        if input_mode == 1:
            # don't change self.action_size as this is used internally in the class
            high = np.array([1] * 1)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        elif input_mode == 2:
            high = np.array([1] * 2)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            high = np.array([1] * 4)
            self._action_space = spaces.Box(-high, high, dtype=np.float323)
        self._input_mode = input_mode
        
    @staticmethod
    def get_pitch_action(action: List[Any], dummy=0) -> np.array:
        """Just use pitch commands"""
        return np.array([dummy, action[0], dummy, dummy])

    @staticmethod
    def get_roll_pitch_action(action: List[Any], dummy=0) -> np.array:
        """Just use roll/pitch commands"""
        return np.array([action[0], action[1], 0, 0])

    def step(self, action: List[Any]) -> GymStepResult:
        if self._input_mode == 1:
            action = CanyonRun.get_pitch_action(action)
        elif self._input_mode == 2:
            action = CanyonRun.get_roll_pitch_action(action)
        # otherwise, assume the input mode is 3
        return super().step(action)