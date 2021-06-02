# AirForceArcade on gym

## Quick start
Install package:
* Go to `airforce-arcade/`
* Run `pip install -e .`

Install unity game:
* Download unity games from [here](https://drive.google.com/drive/folders/1bz4Gp4VG9BX7j7a00iutae1zNnNn4EWG?usp=sharing). 
+ Move downloaded game zip files to `baselines/airforce-arcade/builds/`
+ Unzip games to the `builds/` directory (e.g. `cd builds; unzip DroneDodgeBall.zip`). After decompressing the file, it should look like
```
DroneDodgeBall
      |_ Windows/
      |_ Linux/
      |_ Mac.app/
```
+ Make the game file executable. For Linux:
```
cd builds/DroneDodgeBall/Linux/
chmod -R 755 DroneDodgeBall.x86_64
```


Simple example of how to use the package. More details in [examples/random_action.py](./examples/random_action.py).
```
import gym, airforce_arcade
env = gym.make("DroneDodgeBall-v0")

obs = env.reset()
for i in range(10):
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
env.close()
```

Note that you can send arguments upon environment creation, e.g., 
```
env = gym.make("DroneDodgeBall-v0", uint8_visual=True, use_circle_abstraction=True)
```
Those arguments can be customized in environment class. You might care about those arguments during training and evaluation, which can be set with `env_config` in the .yaml configuration file.

## How to add new environment
You can write your custom environment by basically inherit from `UnityToGymWrapper` class and register to gym.
* Suppose you want to add `CustomEnv` and register it as `CustomEnv-v0` in gym so that you can do `env = gym.make("CustomEnv-v0")`
* In [`./airforce_arcade/__init__.py`](./airforce_arcade/__init__.py), do
```
env_list = [<other envs...>, CustomEnv-v0]

# <other envs registry...>
register(
    id='CustomEnv-v0',
    entry_point='airforce_arcade.envs:CustomEnv', # this should be consistent with your class name
)
```
* In [`./envs/`](./envs/), add your script `customenv.py` that define a class called `CustomEnv` and make sure it's a child class of gym-unity wrapper, i.e., `class CustomEnv(UnityToGymWrapper)`.
* In [`./envs/__init__.py`](./envs/__init__.py), get correct module import path by adding
```
from airforce_arcade.envs.customenv import CustomEnv
```
* Now the only thing left is to define the custom environment, do whatever you want but keep in mind that
  * You need to keep default arguments of gym-unity wrapper, including `uint8_visual`, `flatten_branched`, and `allow_multiple_obs`.
  * You also need to have `worker_id` since this argument specifies the port used by rpc communication between unity standalone instance and python script, and if not being able to be set, you may encounter conflict while starting up multiple environment instances at the same time. 
