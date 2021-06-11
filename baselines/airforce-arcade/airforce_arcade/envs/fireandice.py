import os
import sys
from functools import partial
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from gym import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .multi_agent_gym_unity import UnityToGymWrapper, GymStepResult, MultiAgentGymStepResult, logger
from .env_utils import *


class FireandIce(UnityToGymWrapper):
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
        use_episode_countdown: float = 0,   # greater than 0 for True, otherwise False
        random_seed: float = 0, #0 means let Unity set random seed
    ):
        # Instantiate Unity environment
        dirname = os.path.dirname(__file__)
        if env_build is None:
            if sys.platform == "darwin":
                env_build = os.path.join(dirname, '../../builds/FireandIce/Mac')
            elif sys.platform == 'win32':
                env_build = os.path.join(dirname, '..\\..\\builds\\FireandIce\\Windows\\FireandIce.exe')
            elif sys.platform == 'linux':
                env_build =  os.path.join(dirname, '../../builds/FireandIce/Linux/FireandIce.x86_64')
        self.channel = EnvironmentParametersChannel()
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     no_graphics=no_graphics,
                                     side_channels=[self.channel],
                                     timeout_wait=timeout_wait,
                                     log_folder=log_folder)
        # Set channel params
        self.channel.set_float_parameter('use_episode_countdown', use_episode_countdown)
        self.channel.set_float_parameter('random_seed', random_seed)
        # Instantiate gym-unity
        super(FireandIce, self).__init__(unity_env=unity_env,
                                         uint8_visual=uint8_visual,
                                         flatten_branched=flatten_branched,
                                         allow_multiple_obs=allow_multiple_obs)

        # Modify observation space for reordered raycast and drop ball pose
        self._sensor_obs_shape = (19, 33)
        self._sensor_reorder_mat = generate_reorder_mat(self._sensor_obs_shape[1])
        self.process_sensor_obs = partial(process_raycast, \
            obs_shape=self._sensor_obs_shape, reorder_mat=self._sensor_reorder_mat)

        raw_sensor_obs_shape = 2 * np.prod(self._sensor_obs_shape)
        raw_vec_obs_shape = 9
        for name in self.behavior_name:
            assert self._observation_space[name].shape[0] == (raw_sensor_obs_shape + raw_vec_obs_shape)

            self._observation_space[name] = spaces.Tuple([
                spaces.Box(low=np.zeros(self._sensor_obs_shape, dtype=np.uint8),
                        high=np.ones(self._sensor_obs_shape, dtype=np.uint8) * 255,
                        dtype=np.uint8),
                spaces.Box(low=self._observation_space[name].low[-raw_vec_obs_shape:],
                        high=self._observation_space[name].high[-raw_vec_obs_shape:],
                        dtype=np.float64)
            ])

    def step(self, action: Dict[str, List[Any]]) -> MultiAgentGymStepResult:
        obs, rew, done, info = super().step(action)
        done['__all__'] = np.any(list(done.values()))
        return obs, rew, done, info

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps], name: str) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info, name)

        start, end = 0, np.prod(self._sensor_obs_shape) * 2
        sensor_obs = self.process_sensor_obs(obs[start:end])

        vec_obs = obs[end:]

        obs = [sensor_obs, vec_obs]

        return obs, rew, done, _info

    @property
    def agent_name(self):
        # behavior name is used internally in multi-agent gym unity,
        # whereas agent name allows a better readable naming like red or blue.
        return self.behavior_name
