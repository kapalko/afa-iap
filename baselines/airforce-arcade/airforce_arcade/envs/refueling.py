import os
import sys
from functools import partial
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from gym import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .gym_unity import UnityToGymWrapper, GymStepResult, logger
from .env_utils import *


class Refueling(UnityToGymWrapper):
    def __init__(
        self,
        # [General arguments]
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        no_graphics: bool = False, # turn off rendering
        log_folder: str = None, # directory to logs
        env_build: str = None, # path to Unity build; Set to None to use default relative path
        timeout_wait: int = 600, # timeouot for RPC communication between Unity and python
        # [Game-specific arguments]
        input_mode: float = 3, # 1: thrust only, 2: thrust and roll, 3: thrust and all axes
        randomize_fighter_start_pose: float = 1, # greater than 0 for True, otherwise False
        randomize_fighter_start_thrust: float = 0, # greater than 0 for True, otherwise False
        use_episode_countdown: float = 0,   # greater than 0 for True, otherwise False
        random_seed: float = 0, #0 means let Unity set random seed
    ):
        # Instantiate Unity environment
        dirname = os.path.dirname(__file__)
        if env_build is None:
            if sys.platform == "darwin":
                env_build = os.path.join(dirname, '../../builds/Refueling/Mac')
            elif sys.platform == 'win32':
                env_build = os.path.join(dirname, '..\\..\\builds\\Refueling\\Windows\\Refueling.exe')
            elif sys.platform == 'linux':
                env_build =  os.path.join(dirname, '../../builds/Refueling/Linux/Refueling.x86_64')
        self.channel = EnvironmentParametersChannel()
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[self.channel],
                                     timeout_wait=timeout_wait,
                                     log_folder=log_folder)

        # Set channel params
        self.channel.set_float_parameter('input_mode', input_mode)
        self.channel.set_float_parameter('randomize_fighter_start_pose', randomize_fighter_start_pose)
        self.channel.set_float_parameter('randomize_fighter_start_thrust', randomize_fighter_start_thrust)
        self.channel.set_float_parameter('use_episode_countdown', use_episode_countdown)
        self.channel.set_float_parameter('random_seed', random_seed)

        # Instantiate gym-unity
        super(Refueling, self).__init__(unity_env=unity_env,
                                        uint8_visual=uint8_visual,
                                        flatten_branched=flatten_branched,
                                        allow_multiple_obs=allow_multiple_obs)

        # Modify action space based on input mode
        if flatten_branched:
            if input_mode == 1:
                self._action_space = spaces.Discrete(3)
            elif input_mode == 2:
                self._action_space = spaces.Discrete(6)
        else:
            if input_mode == 1:
                self._action_space = spaces.MultiDiscrete([3])
            elif input_mode == 2:
                self._action_space = spaces.MultiDiscrete([3, 3])
        self._input_mode = input_mode

        # Modify observation space for reordered raycast and drop ball pose
        self._sensor_obs_shape = (19, 33)
        self._sensor_reorder_mat = generate_reorder_mat(self._sensor_obs_shape[1])
        self.process_sensor_obs = partial(process_raycast, \
            obs_shape=self._sensor_obs_shape, reorder_mat=self._sensor_reorder_mat)

        raw_sensor_obs_shape = 2 * np.prod(self._sensor_obs_shape)
        raw_vec_obs_shape = 11
        self._waypoint_idcs = range(-raw_vec_obs_shape, -raw_vec_obs_shape + 3)
        self._ballpose_idcs = range(-raw_vec_obs_shape + 3, -raw_vec_obs_shape + 6)
        assert self._observation_space.shape[0] == (raw_sensor_obs_shape + raw_vec_obs_shape)

        self._observation_space = spaces.Tuple([
            spaces.Box(low=np.zeros(self._sensor_obs_shape, dtype=np.uint8),
                       high=np.ones(self._sensor_obs_shape, dtype=np.uint8) * 255,
                       dtype=np.uint8),
            spaces.Box(low=self._observation_space.low[-raw_vec_obs_shape:],
                       high=self._observation_space.high[-raw_vec_obs_shape:],
                       dtype=np.float64)
        ])

    @staticmethod
    def get_pitch_action(action: List[Any], dummy=1) -> np.array: # action is 0, 1, 2; 1 is neutral
        """Just use pitch commands"""
        return np.array([dummy, action[0], dummy, dummy])

    @staticmethod
    def get_pitch_throttle_action(action: List[Any], dummy=1) -> np.array:
        """Just use pitch/throttle commands"""
        return np.array([dummy, action[0], dummy, action[1]])

    def step(self, action: List[Any]) -> GymStepResult:
        # action in unity: (roll, pitch, yaw, throttle)
        if self._input_mode == 1: # pitch only
            action = self.get_pitch_action(action)
        elif self._input_mode == 2: # pitch+throttle
            action = self.get_pitch_throttle_action(action)
        # otherwise, assume the input mode is 3
        return super().step(action)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)

        _info['waypoint'] = obs[self._waypoint_idcs]
        _info['ball'] = obs[self._ballpose_idcs]

        start, end = 0, np.prod(self._sensor_obs_shape) * 2
        sensor_obs = self.process_sensor_obs(obs[start:end])

        vec_obs = obs[end:]

        obs = [sensor_obs, vec_obs]

        return obs, rew, done, _info
