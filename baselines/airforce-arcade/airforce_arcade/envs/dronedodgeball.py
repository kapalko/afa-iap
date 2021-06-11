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


class DroneDodgeBall(UnityToGymWrapper):
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
        use_held_ball: float = 0, # 0: ball launcher, 1: held ball, 2: set abs. position, 3: set rel. position
        num_launchers: float = 1,
        min_ball_launcher_force: float = 60.,
        max_ball_launcher_force: float = 90.,
        apf_human_mixing_fraction: float = 0.0,
        forest_density: float = 0.0, # forest density from 0 to 1
        forest_velocity: float = 0.0, # positive or negative float, 0 (not moving) by default
        teleporting_trees: float = 0, # greater than 0 as True, otherwise False
        teleport_tree_probability: float = 0.0, # from 0 to 1
        logging_episode_summaries: float = 0, # 0 or 1
        capture_frame_buffer: float = 0,
        capture_frame_fps: float = 50,
        use_episode_countdown: float = 0,   # greater than 0 for True, otherwise False
        random_seed: float = 0, #0 means let Unity set random seed
    ):
        # Check argument
        if capture_frame_buffer:
            assert not no_graphics, "no_graphics should be False otherwise you will get black image."
            assert log_folder is not None, "You need to specify log_folder if capture_frame_buffer set to true."
            assert os.path.isabs(log_folder), "log_folder should be absolute path."
            if not os.path.isdir(log_folder):
                os.makedirs(log_folder)

        # Instantiate Unity environment
        dirname = os.path.dirname(__file__)
        if env_build is None:
            if sys.platform == "darwin":
                env_build = os.path.join(dirname, '../../builds/DroneDodgeBall/Mac')
            elif sys.platform == 'win32':
                env_build = os.path.join(dirname, '..\\..\\builds\\DroneDodgeBall\\Windows\\DroneDodgeBall.exe')
            elif sys.platform == 'linux':
                env_build =  os.path.join(dirname, '../../builds/DroneDodgeBall/Linux/DroneDodgeBall.x86_64')
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
        self.channel.set_float_parameter('use_held_ball', use_held_ball)
        self.channel.set_float_parameter('num_launchers', num_launchers)
        self.channel.set_float_parameter('min_ball_launcher_force', min_ball_launcher_force)
        self.channel.set_float_parameter('max_ball_launcher_force', max_ball_launcher_force)
        self.channel.set_float_parameter('apf_human_mixing_fraction', apf_human_mixing_fraction)
        self.channel.set_float_parameter('forest_density', forest_density)
        self.channel.set_float_parameter('forest_velocity', forest_velocity)
        self.channel.set_float_parameter('teleporting_trees', teleporting_trees)
        self.channel.set_float_parameter('teleport_tree_probability', teleport_tree_probability)
        self.channel.set_float_parameter('logging_episode_summaries', logging_episode_summaries)
        self.channel.set_float_parameter('capture_frame_buffer', capture_frame_buffer)
        self.channel.set_float_parameter('capture_frame_fps', capture_frame_fps)
        self.channel.set_float_parameter('use_episode_countdown', use_episode_countdown)
        self.channel.set_float_parameter('random_seed', random_seed)

        # Instantiate gym-unity
        super(DroneDodgeBall, self).__init__(unity_env=unity_env,
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
        raw_vec_obs_shape = 10
        assert self._observation_space.shape[0] == (raw_sensor_obs_shape + raw_vec_obs_shape)

        self._observation_space = spaces.Tuple([
            spaces.Box(low=np.zeros(self._sensor_obs_shape, dtype=np.uint8),
                       high=np.ones(self._sensor_obs_shape, dtype=np.uint8) * 255,
                       dtype=np.uint8),
            spaces.Box(low=self._observation_space.low[-raw_vec_obs_shape:],
                       high=self._observation_space.high[-raw_vec_obs_shape:],
                       dtype=np.float64)
        ])

    def step(self, action: List[Any]) -> GymStepResult:
        dummy = 0
        if self._input_mode == 1:
            action = np.array([dummy, dummy, action[0], dummy]) # thrust only (up/down)
        elif self._input_mode == 2:
            action = np.array([dummy, action[0], action[1], dummy]) # roll + thrust
        # full control: pitch / roll / thrust / yaw

        return super().step(action)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)

        start, end = 0, np.prod(self._sensor_obs_shape) * 2
        sensor_obs = self.process_sensor_obs(obs[start:end])

        vec_obs = obs[end:]

        obs = [sensor_obs, vec_obs]

        return obs, rew, done, _info
