import os
import sys
from functools import partial
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

from .gym_unity import UnityToGymWrapper, GymStepResult, logger
from .env_utils import *


class TimedWaypoints(UnityToGymWrapper):
    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        no_graphics: bool = False,
        log_folder: str = None,
        min_number_of_exclusion_zones: int = 2,
        max_number_of_exclusion_zones: int = 6,
        min_waypoint_value: int = 1,
        max_waypoint_value: int = 5,
        min_time_until_scorable: int = 10,
        max_time_until_scorable: int = 20,
        min_scorable_duration: int = 10,
        max_scorable_duration: int = 40,
        time_between_waypoint_spawn: int = 10,
        max_number_waypoints: int = 10,
        tanker_refueling_distance: float = 100.,
        tanker_refueling_ratio: float = 100.,
        game_space_norm: float = 256.,
        game_xmax: float = 256.,
        game_ymax: float = 128.,
        game_zmax: float = 256.,
        env_build: str = None,
        set_terrain: bool = False,
        timeout_wait: int = 600,
        use_episode_countdown: float = 0,   # greater than 0 for True, otherwise False
        random_seed: float = 0, #0 means let Unity set random seed
    ):
        dirname = os.path.dirname(__file__)
        if env_build is None:
            if sys.platform == "darwin":
                env_build = os.path.join(dirname, '../../builds/TimedWaypoints/Mac')
            elif sys.platform == 'win32':
                env_build = os.path.join(dirname, '..\\..\\builds\\TimedWaypoints\\Windows\\TimedWaypoints.exe')
            elif sys.platform == 'linux':
                env_build =  os.path.join(dirname, '../../builds/TimedWaypoints/Linux/TimedWaypoints.x86_64')
        self.channel = EnvironmentParametersChannel()
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[self.channel],
                                     timeout_wait=timeout_wait,
                                     log_folder=log_folder)
        self.channel.set_float_parameter("min_number_of_exclusion_zones", float(min_number_of_exclusion_zones))
        self.channel.set_float_parameter("max_number_of_exclusion_zones", float(max_number_of_exclusion_zones))
        self.channel.set_float_parameter("min_waypoint_value", float(min_waypoint_value))
        self.channel.set_float_parameter("max_waypoint_value", float(max_waypoint_value))
        self.channel.set_float_parameter("min_time_until_scorable", float(min_time_until_scorable))
        self.channel.set_float_parameter("max_time_until_scorable", float(max_time_until_scorable))
        self.channel.set_float_parameter("min_scorable_duration", float(min_scorable_duration))
        self.channel.set_float_parameter("max_scorable_duration", float(max_scorable_duration))
        self.channel.set_float_parameter("time_between_waypoint_spawn", float(time_between_waypoint_spawn))
        self.channel.set_float_parameter("max_number_waypoints", float(max_number_waypoints))
        self.channel.set_float_parameter("max_time_until_scorable", float(max_time_until_scorable))
        self.channel.set_float_parameter("tanker_refueling_distance", tanker_refueling_distance)
        self.channel.set_float_parameter("tanker_refueling_ratio", tanker_refueling_ratio)
        self.channel.set_float_parameter("game_space_norm", game_space_norm)
        self.channel.set_float_parameter("game_xmax", game_xmax)
        self.channel.set_float_parameter("game_xmin", -game_xmax)
        self.channel.set_float_parameter("game_ymax", game_ymax)
        self.channel.set_float_parameter("game_ymin", 0)
        self.channel.set_float_parameter("game_zmax", game_zmax)
        self.channel.set_float_parameter("game_zmin", -game_zmax)
        self.channel.set_float_parameter('use_episode_countdown', use_episode_countdown)
        set_terrain = 1. if set_terrain else 0.
        self.channel.set_float_parameter("set_terrain", set_terrain)
        self.channel.set_float_parameter('random_seed', random_seed)

        super(TimedWaypoints, self).__init__(unity_env=unity_env,
                                             uint8_visual=uint8_visual,
                                             flatten_branched=flatten_branched,
                                             allow_multiple_obs=allow_multiple_obs)

        self._sensor_obs_shape = (19, 33)
        self._sensor_reorder_mat = generate_reorder_mat(self._sensor_obs_shape[1])
        self.process_sensor_obs = partial(process_raycast, obs_shape=self._sensor_obs_shape, reorder_mat=self._sensor_reorder_mat)

        raw_sensor_obs_shape = 2 * np.prod(self._sensor_obs_shape)
        raw_vec_obs_shape = 11
        assert self._observation_space.shape[0] == (raw_sensor_obs_shape + raw_vec_obs_shape)

        self._observation_space = spaces.Tuple([spaces.Box(low=np.zeros(self._sensor_obs_shape, dtype=np.uint8),
                high=np.ones(self._sensor_obs_shape, dtype=np.uint8) * 255,
                dtype=np.uint8), spaces.Box(low=self._observation_space.low[-raw_vec_obs_shape:],
                high=self._observation_space.high[-raw_vec_obs_shape:],
                dtype=np.float64)])

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)

        start, end = 0, np.prod(self._sensor_obs_shape) * 2
        sensor_obs = self.process_sensor_obs(obs[start:end])

        vec_obs = obs[end:]

        obs = [sensor_obs, vec_obs]

        return obs, rew, done, _info
