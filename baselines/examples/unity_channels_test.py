#test switching between Ball Launcher and ICRA-Style Environments
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

env_build_file = 'DroneDodgeball12.4'

channel = EnvironmentParametersChannel()
unity_env = UnityEnvironment(env_build_file, side_channels=[channel])
print("Ball Launcher Env")
channel.set_float_parameter("useHeldBall", -1.0)
env = UnityToGymWrapper(unity_env)
obs = env.reset()
done = False
for i in range(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break
print("ICRA-Style Env")
channel.set_float_parameter("useHeldBall", 1.0)
env = UnityToGymWrapper(unity_env)
obs = env.reset()
done = False
for i in range(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break
env.close()
