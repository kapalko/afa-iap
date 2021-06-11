import inspect
from functools import partial
import itertools
import numpy as np
from typing import Any, Dict, List, Tuple, Union

import gym
from gym import error, spaces

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)

GymStepResult = Tuple[np.ndarray, float, bool, Dict]
MultiAgentGymStepResult = Tuple[Dict[str, np.ndarray], \
    Dict[str, float], Dict[str, bool], Dict[str, Dict]]


def dictOfList2listOfDict(dictOfList):
    return [dict(zip(dictOfList, i)) for i in zip(*dictOfList.values())]


class UnityToGymWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    """

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        self.name = list(self._env.behavior_specs.keys())
        self.group_spec = {name: self._env.behavior_specs[name] for name in self.name}

        if np.any([self._get_n_vis_obs(v) == 0 for v in self.name]) and \
            np.any([self._get_vec_obs_size(v) == 0 for v in self.name]):
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not np.any([self._get_n_vis_obs(v) >= 1 for v in self.name]) and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if (
            np.any([self._get_n_vis_obs(v) > 0 and self._get_vec_obs_size(v) > 0 for v in self.name])
            and not self._allow_multiple_obs
        ):
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        decision_steps = self._multi_agent_call(self._env.get_steps, {n: n for n in self.name}, 0)
        self._previous_decision_step = decision_steps

        # Set action spaces
        self.action_size = dict()
        self._action_space = dict()
        self._flattener = {n: None for n in self.name}
        for name, group_spec in self.group_spec.items():
            if group_spec.action_spec.is_discrete():
                self.action_size[name] = group_spec.action_spec.discrete_size
                branches = group_spec.action_spec.discrete_branches
                if group_spec.action_spec.discrete_size == 1:
                    self._action_space[name] = spaces.Discrete(branches[0])
                else:
                    if flatten_branched:
                        self._flattener[name] = ActionFlattener(branches)
                        self._action_space[name] = self._flattener[name].action_space
                    else:
                        self._action_space[name] = spaces.MultiDiscrete(branches)

            elif group_spec.action_spec.is_continuous():
                if flatten_branched:
                    logger.warning(
                        "The environment has a non-discrete action space. It will "
                        "not be flattened."
                    )

                self.action_size[name] = group_spec.action_spec.continuous_size
                high = np.array([1] * group_spec.action_spec.continuous_size)
                self._action_space[name] = spaces.Box(-high, high, dtype=np.float32)

            else:
                raise UnityGymException(
                    "The gym wrapper does not provide explicit support for both discrete "
                    "and continuous actions."
                )

        # Set observations space
        self._observation_space = dict()
        for name in self.name:
            list_spaces: List[gym.Space] = []
            for shape in self._get_vis_obs_shape(name):
                if uint8_visual:
                    list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
                else:
                    list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
            if self._get_vec_obs_size(name) > 0:
                # vector observation is last
                high = np.array([np.inf] * self._get_vec_obs_size(name))
                list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
            if self._allow_multiple_obs:
                self._observation_space[name] = spaces.Tuple(list_spaces)
            else:
                self._observation_space[name] = list_spaces[0]  # only return the first one

    def reset(self) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (dict of object/list): the initial observation of the
        space.
        """
        if not self.game_over:
            for i in range(len(self.name)):
                self._env.reset()
        decision_step, terminal_step = dictOfList2listOfDict(self._multi_agent_call(\
            self._env.get_steps, {n: n for n in self.name}))
        self.game_over = False
        res = self._multi_agent_call(self._single_step, decision_step, 0)
        return res

    def step(self, actions: Dict[str, List[Any]]) -> MultiAgentGymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (dict of object/list): agent's observation of the current environment
            reward (dict of float/list) : amount of reward returned after previous action
            done (dict of boolean/list): whether the episode has ended.
            info (dict of dict): contains auxiliary diagnostic information.
        """
        for name in self.name:
            action = actions[name]
            if self._flattener[name] is not None:
                # Translate action into list
                action = self._flattener[name].lookup_action(action)

            action = np.array(action).reshape((1, self.action_size[name]))

            action_tuple = ActionTuple()
            if self.group_spec[name].action_spec.is_continuous():
                action_tuple.add_continuous(action)
            else:
                action_tuple.add_discrete(action)
            self._env.set_actions(name, action_tuple)

        self._env.step() # NOTE: step only happens when all actions set

        decision_step, terminal_step = dictOfList2listOfDict(self._multi_agent_call(\
            self._env.get_steps, {n: n for n in self.name}))

        if np.any([len(v) != 0 for v in terminal_step.values()]):
            # The agent is done
            self.game_over = True
            return dictOfList2listOfDict(self._multi_agent_call(self._single_step, terminal_step))
        else:
            return dictOfList2listOfDict(self._multi_agent_call(self._single_step, decision_step))

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps], name: str) -> GymStepResult:
        if self._allow_multiple_obs:
            visual_obs = self._get_vis_obs_list(info)
            visual_obs_list = []
            for obs in visual_obs:
                visual_obs_list.append(self._preprocess_single(obs[0]))
            default_observation = visual_obs_list
            if self._get_vec_obs_size(name) >= 1:
                default_observation.append(self._get_vector_obs(info)[0, :])
        else:
            if self._get_n_vis_obs(name) >= 1:
                visual_obs = self._get_vis_obs_list(info)
                default_observation = self._preprocess_single(visual_obs[0][0])
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        if self._get_n_vis_obs(name) >= 1:
            visual_obs = self._get_vis_obs_list(info)
            self.visual_obs = self._preprocess_single(visual_obs[0][0])

        done = isinstance(info, TerminalSteps)

        return (default_observation, info.reward[0], done, {"step": info})

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _multi_agent_call(self, fn, inps, idcs=None):
        out = dict()
        fn_args = inspect.getfullargspec(fn).args
        use_partial_fn = 'name' in fn_args
        for name, inp in inps.items():
            partial_fn = partial(fn, name=name) if use_partial_fn else fn
            res = partial_fn(inp)
            if idcs is None:
                out[name] = res
            elif isinstance(idcs, int):
                out[name] = res[idcs]
            elif isinstance(idcs, list):
                out[name] = [res[idx] for idx in idcs]
            else:
                raise ValueError('idcs should be None, an int, or a list.')
        return out

    def _get_n_vis_obs(self, name: str) -> int:
        result = 0
        for obs_spec in self.group_spec[name].observation_specs:
            if len(obs_spec.shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self, name: str) -> List[Tuple]:
        result: List[Tuple] = []
        for obs_spec in self.group_spec[name].observation_specs:
            if len(obs_spec.shape) == 3:
                result.append(obs_spec.shape)
        return result

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self, name) -> int:
        result = 0
        for obs_spec in self.group_spec[name].observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment %s", self.name)
        return

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def behavior_name(self):
        return self.name


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]
