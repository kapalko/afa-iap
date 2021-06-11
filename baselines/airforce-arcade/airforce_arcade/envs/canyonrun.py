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


class CanyonRun(UnityToGymWrapper):
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
        timeout_wait: int = 600, # timeout for RPC communication between Unity and python
        # [Game-specific arguments]
        input_mode: float = 3, # (1) pitch-control only environment/straight mode (2) pitch, roll, (3) pitch, roll, yaw, and throttle
        straight_mode: float = 1, #1 = straight mode; 0 = turns
        terrain_mode: int = 0, #0=regular, 1 = narrow, 2= gaps
        max_jet_height: float = 250, #max height the jet can reach before episode ends
        waypoint_min_height_ratio: float = 0.4, # ratio of max_jet_height to the min waypoint height
        waypoint_max_height_ratio: float = 0.8, # ratio of max_jet_height to the max waypoint_height
        randomize_start_height: float = 1, # whether to randomize the height of the jet on episode begin
        use_launchers: float = 0, # whether to shoot at the jet with ball launchers
        use_horizontal_shift: float = 1, # whether to horizontally randomly shift waypoint
        waypoint_max_horizontal_shift_ratio: float = 0.05,
        survival_reward: float = 0, # reward each frame for not dying
        waypoint_reward: float = 0.5, # reward for collecting waypoints
        logging_episode_summaries: float = 0, # 0 or 1
        extensive_logging: float = 0, # 0 or 1; whether to write info for each episode in Debug.Log
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
                env_build = os.path.join(dirname, '../../builds/CanyonRun/Mac')
            elif sys.platform == 'win32':
                env_build = os.path.join(dirname, '..\\..\\builds\\CanyonRun\\Windows\\CanyonRun.exe')
            elif sys.platform == 'linux':
                env_build =  os.path.join(dirname, '../../builds/CanyonRun/Linux/CanyonRun.x86_64')
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
        self.channel.set_float_parameter('straight_mode', straight_mode)
        self.channel.set_float_parameter('terrain_mode', terrain_mode)
        self.channel.set_float_parameter('max_jet_height', max_jet_height)
        self.channel.set_float_parameter('waypoint_min_height_ratio', waypoint_min_height_ratio)
        self.channel.set_float_parameter('waypoint_max_height_ratio', waypoint_max_height_ratio)
        self.channel.set_float_parameter('randomize_start_height', randomize_start_height)
        self.channel.set_float_parameter('use_launchers', use_launchers)
        self.channel.set_float_parameter('use_horizontal_shift', use_horizontal_shift)
        self.channel.set_float_parameter('waypoint_max_horizontal_shift_ratio', waypoint_max_horizontal_shift_ratio)
        self.channel.set_float_parameter('logging_episode_summaries', logging_episode_summaries)
        self.channel.set_float_parameter('extensive_logging', extensive_logging)
        self.channel.set_float_parameter('capture_frame_buffer', capture_frame_buffer)
        self.channel.set_float_parameter('capture_frame_fps', capture_frame_fps)
        self.channel.set_float_parameter('use_episode_countdown', use_episode_countdown)
        self.channel.set_float_parameter('waypoint_reward', waypoint_reward)
        self.channel.set_float_parameter('survival_reward', survival_reward)
        self.channel.set_float_parameter('random_seed', random_seed)

        # Instantiate gym-unity
        super(CanyonRun, self).__init__(unity_env=unity_env,
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

        self._sensor_obs_shape = (19, 33)
        self._sensor_reorder_mat = generate_reorder_mat(self._sensor_obs_shape[1])
        self.process_sensor_obs = partial(process_raycast, \
            obs_shape=self._sensor_obs_shape, reorder_mat=self._sensor_reorder_mat)

        raw_sensor_obs_shape = 2 * np.prod(self._sensor_obs_shape)
        raw_vec_obs_shape = 7
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
    def get_pitch_action(action: List[Any], dummy=0) -> np.array:
        """Just use pitch commands"""
        return np.array([dummy, action[0], dummy, dummy])

    @staticmethod
    def get_roll_pitch_action(action: List[Any], dummy=0) -> np.array:
        """Just use roll/pitch commands"""
        return np.array([action[0], action[1], 0, 0])

    def step(self, action: List[Any]) -> GymStepResult:
        # action in unity: (roll, pitch, yaw, throttle)
        if self._input_mode == 1: # pitch only
            action = CanyonRun.get_pitch_action(action)
        elif self._input_mode == 2: # roll+pitch
            action = CanyonRun.get_roll_pitch_action(action)
        # otherwise, assume the input mode is 3
        return super().step(action)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)

        start, end = 0, np.prod(self._sensor_obs_shape) * 2
        sensor_obs = self.process_sensor_obs(obs[start:end])

        vec_obs = obs[end:]

        obs = [sensor_obs, vec_obs]

        return obs, rew, done, _info
