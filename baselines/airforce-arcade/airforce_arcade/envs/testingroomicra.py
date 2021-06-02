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

from .gym_unity import UnityToGymWrapper, GymStepResult


class TestingRoomICRA(UnityToGymWrapper):
    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 0,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        use_circle_abstraction: bool = False
    ):
        dirname = os.path.dirname(__file__)
        if sys.platform == "darwin":
            env_build = os.path.join(dirname, '../../builds/TestingRoomICRA/Mac')
        elif sys.platform == 'win32':
            env_build = os.path.join(dirname, '..\\..\\builds\\TestingRoomICRA\\Windows\\ICRATestingRoom.exe')
        elif sys.platform == 'linux':
            env_build =  os.path.join(dirname, '../../builds/TestingRoomICRA/Linux/TestingRoomICRA_Linux.x86_64')
        unity_env = UnityEnvironment(env_build,
                                     seed=seed,
                                     worker_id=worker_id,
                                     side_channels=[])
        super(TestingRoomICRA, self).__init__(unity_env=unity_env,
                                              uint8_visual=uint8_visual,
                                              flatten_branched=flatten_branched,
                                              allow_multiple_obs=allow_multiple_obs)

        self._use_circle_abstraction = use_circle_abstraction
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0., 1., dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size()).astype(np.float32)
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._use_circle_abstraction:
            high = np.array([np.inf] * 4).astype(np.float32)
            list_spaces[0] = spaces.Box(-high, high, dtype=np.float32)
        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        (obs, rew, done, _info) = super()._single_step(info)
        if self._use_circle_abstraction:
            assert obs[0].dtype == np.uint8 and len(obs[0].shape) == 3
            obs[0] = self._hough_circles(obs[0])

        return (obs, rew, done, _info)

    def _hough_circles(self, img):
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 
                                   dp = 1, 
                                   minDist = 100, 
                                   param1 = 30, 
                                   param2 = 10, 
                                   minRadius = 1, 
                                   maxRadius = 100)
        if circles is not None: # assumes there is only one circle!
            circles = circles[0, 0]
            x = circles[0]
            y = circles[1]
            x_pix = min(np.uint16(np.around(x)), img.shape[1]-1)
            y_pix = min(np.uint16(np.around(y)), img.shape[0]-1)
            z = img[y_pix, x_pix] # depth at center of detected circle
            return [x, y, circles[2], z]
        else:
            return -np.ones((4,))
