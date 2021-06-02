import gym
import airforce_arcade

env = gym.make("TestingRoomICRA-v0", uint8_visual=True, allow_multiple_obs=True,
               use_circle_abstraction=True)
# env = gym.make("DroneDodgeBall-v0", uint8_visual=True, use_ball_pose=True, no_graphics=True)
# env = gym.make("Refueling-v0", no_graphics=True)
# env = gym.make("TimedWaypoints-v0", no_graphics=True)

for eps_i in range(10):
    obs = env.reset()
    done = False
    total_rew = 0
    while not done:
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        total_rew += rew
    print("[Episode {}] Reward: {}".format(eps_i, total_rew))
