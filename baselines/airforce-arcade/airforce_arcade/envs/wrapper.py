import gym
import itertools
import numpy as np


class FlattenActionWrapper(gym.ActionWrapper):
    """Wrap MultiDiscrete action space to Discrete action space."""
    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        nvec = env.action_space.nvec
        ndim = np.prod(nvec)
        self.action_space = gym.spaces.Discrete(ndim) # don't affect wrapped class, i.e., env.action_space
        self.prods = [1] + list(itertools.accumulate(nvec[1:], lambda x, y: x * y))
        self.prods.reverse()

    def action(self, action):
        out = []
        for prod in self.prods:
            out.append(action // prod)
            action = action % prod
        return out
