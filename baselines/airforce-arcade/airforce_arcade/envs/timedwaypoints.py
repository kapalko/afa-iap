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


class TimedWaypoints(UnityToGymWrapper):
    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        stacked_vec_obs: int = 1,
        no_graphics: bool = False,
        log_folder: str = None
    ):
        dirname = os.path.dirname(__file__)
        if sys.platform == "darwin":
            env_build = os.path.join(dirname, '../../builds/TimedWaypoints/Mac')
        elif sys.platform == 'win32':
            env_build = os.path.join(dirname, '..\\..\\builds\\TimedWaypoints\\Windows\\TimedWaypoints.exe')
        elif sys.platform == 'linux':
            env_build =  os.path.join(dirname, '../../builds/TimedWaypoints/Linux/TimedWaypoints.x86_64')
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[],
                                     log_folder=log_folder)
        super(TimedWaypoints, self).__init__(unity_env=unity_env,
                                             uint8_visual=uint8_visual,
                                             flatten_branched=flatten_branched,
                                             allow_multiple_obs=allow_multiple_obs,
                                             stacked_vec_obs=stacked_vec_obs)
