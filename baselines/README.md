# Training agent with RLLib

## Quick Start
**Run training (remember to set `num_worker` in .yaml file).** You can find copied configuration, training log, checkpoint, etc in `--local-dir`.
```
python train.py -f ./config/dodgeball-ppo-fcnet.yaml
                --local-dir <root-dir-to-save-training-log>
                --temp-dir <root-dir-to-save-ray-session-log>
                --checkpoint-freq 10
                --checkpoint-at-end
```
Check [`./config/dodgeball-ppo-convnet.yaml`](./config/dronedodgeball/dronedodgeball-ppo-convnet.yaml) for more details. Note that `--ray-num-cpus`, `--ray-num-gpus`, `--ray-num-nodes`, and `--ray-object-store-memory` are for ray server, which is lower-level than `num_gpus`, `num_cpus`, etc specified in .yaml file. Normally, you only need to change `num_gpus` and `num_workers` in the configuration file. Also refer to [common configs](https://docs.ray.io/en/master/rllib-training.html#common-parameters) and [ppo-specific configs](https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo) for more details.

**Run evaluation.** The path to checkpoint look something like `<exp-name>/<trainer>_<env>_<timestamp>/checkpoint_<N>/checkpoint-<N>`.
```
python rollout.py <path-to-checkpoint> --run PPO --episodes 1 --num-workers 0 --num-gpus 0
```
This will dump a `rollout` file (and `monitor/` if use `--monitor`) in `<exp-name>/<trainer>_<env>_<timestamp>/checkpoint_<N>/`. The `rollout` file contains experience (and `info` if use `--save-info`) of all episode run and `monitor/` contains video. You can use these logs to do metric evaluation, get statistics, visualization, etc.

## Setup
* Follow [`./airforce-arcade/README.md`](./airforce-arcade/README.md) to setup environment.
* Dependencies
  * `tensorflow` or `tensorflow-gpu`
  * `opencv-python`
  * `gym`
  * `matplotlib`
  * `ray[rllib]`

## How to add new model
* Add `customnet.py` in `./model/`
* Class name MUST be `CustomModel` and the class should inherit from `TFModelV2` or its children class like `RecurrentNetwork`, etc.
* Add corresponding configuration file, e.g., `dodgeball-ppo-customnet.yaml` in `./config/`. Make sure you have:
  * Correct experiment name (the first line of the .yaml)
  * Correct model name
    ```
    <exp-name>:
      <other-specs...>:
      config:
        <other-specs...>:
          model:
            custom_model: customnet # this should be the same as your python file name (w/o .py)
            custom_model_config:
              <whatever-custom-arguments-of-model-class>      
    ```
* Note that your model have arguments including `obs_space`, `action_space`, `num_outputs`, `model_config`, and `name` (due to inheritance from `TFModelV2`). All other custom arguments can be simply added to class instantiation (`__init__`) but make sure they are the same as what you specify in `custom_model_config` above.

## Tips
* For debugging, set `num_worker=0` in the .yaml file and use `--local-mode` when running python, since by default, the code run with remote worker(s).
* Set `train_batch_size` and `sgd_minibatch_size` in .yaml file to a much smaller number, e.g., 2 when debugging. It allows you to spend much less time on off-policy data collection and test whether there is bug in an entire training iteration.
* On Supercloud, you may want to make sure your script can be successfully run in interactive job before submitting as batch jobs.
* Most failures result from failed X server, which is required to run Unity standalone/build. There are several commands for checking whether X server is correctly running,
  * `xset -q`. You should see `DPMS is Enabled` and `Monitor On` at the end.
  * `glxinfo`. You should see a lot of messages related to OpenGL. Also you can check OpenGL version by running `glxinfo | grep "OpenGL version"`, where OpenGL 3.2+ is required for Unity.
  * `glxgears`. You should see something like `172412 frames in 5.0 seconds = 34482.309 FPS`.
* Some nodes on the cluster is not correctly configured such that we cannot successfully run our job. A hacky way to inspect the node is to directly connect to the node (BE VERY CAUTIAUS about doing so as access to a node should only be granted by job scheduler. Always connect ONLY to jobs assigned to you.) 
  * `LLstat`. Get node identifier from `NODELIST`. It looks something like `d-9-14-2`.
  * From your login node, `ssh d-x-x-x`.
  * You can do something like `nvidia-smi` or `ps aux | grep xxx`, etc.
* On Supercloud, sometimes you will ssh to a different login node, you can then do `ssh login-X # X can be 2,3,4` after login to switch to the node you want. Note that if you do port forwarding at a node different from the one you run tensorboard, then you cannot see it.
* Using Tensorboard.
  * In login node (to my knowledge, we cannot do this on computation nodes since we are not able to directly connect to them without hop), run 
  ```
  export TMPDIR=~/tmp; tensorboard --logdir . --port 6006 --bind_all # otherwise permission error because of attempt to access /tmp
  ```
  * Do port forwarding from your local machine, run
  ```
  ssh <user-name>@<supercloud> -L 127.0.0.1:16006:127.0.0.1:6006 # <client localhost with port>:<server localhost with port>
  ``` 
  Remember to make sure you are doing port forwarding to the same login node as you running Tensorboard.

## Known Issues
* Unity standalones/builds use a lot of threads (hundreds). Each worker in RLLib normally corresponds to an agent collecting data in one environment and thus you are likely to get into thread issue if using too many workers. There are several ways to handle thread issues.
    * You can check number of threads used in a process by running `ps -o nlwp <pid>`; note that your `<pid>` should be the one of running your unity standalone instead of your python script since we start unity game with subprocess call.
    * Change all rendering related parameters to low when you are building Unity standalone, e.g., `Project Settings --> Graphics`, `Edit --> Graphic Tier`, `Window --> Rendering --> Lighting Settings`, etc.
    * If the environment does not require visual observation, or depth map in our case, you can set `no_graphics` in `env_config` in configuration `.yaml` file. This also makes training much faster. Note that before doing this, please check the next bullet point.
* Current implementation on Unity side does not allow turning on/off camera sensor, which requires rendering pipeline, and thus if we set `no_graphics` for those environments, we will get `Requested RenderTexture format RGBA4 (142) is not supported on this platform, using RGBA8 UNorm (8) fallback format` for EVERY time step. This will make your log file super large so NEVER do this in batch job as our current launch sequence . But it's probably okay to do it with interactive job since by default we don't dump log file for Unity if `log_folder` is not set in `env_config`. 
* For `rollout.py`, `--monitor` argument does not work since this requires getting visual observation from the environment and the only environment we have now that supports visual observation is `DroneDodgeBall`, whose visual observation is depth map and the `--monitor` argument will record video of depth map instead of RGB frames. The way I've been recording videos is do screen recording directly on the game play, which sidesteps communication between unity game and python script.
* There is a bug running LSTM with multi-GPU setting. The error message is `Error: Cannot merge devices with incompatible ids: '/job:localhost/replica:0/task:0/device:GPU:0' and '/job:localhost/replica:0/task:0/device:GPU:1'`. A related issue is shown [here](https://github.com/ray-project/ray/issues/7747). This may not be a big deal for now as we are mostly working on vector observation and don't really need GPU for agent learning.
* Some nodes on Supercloud are not correctly configured for X server. Instead of using GPU for X server by runnning `startx &`, we can use CPU with `xvfb`. You can either 
  * Start X session on a display by 
  ```
  Xvfb :99 &
  export DISPLAY=:99
  ``` 
  * Or launch per comand line call by 
  ```
  xvfb-run --auto-servernum --server-args='-screen 0 640x480x24 python xxx.py
  ``` 
  More details in [here](http://elementalselenium.com/tips/38-headless) and you can find this in [train_template.sh](./LLSC/train_template.sh). Note that this does not support graphic mode in Unity as Unity requires OpenGL 3.2+ while `mesa` on Supercloud only support OpenGL 3.1.0 (you can check this by running `xfvb-run --auto-servernum glxinfo | grep "OpenGL version"`). We need to update from `mesa 20.0.8` to a higher version and I haven't figured out how to do this without sudo privilege. There are some discussion on using CPU rendering (xvfb) for MLAgents in [here](https://github.com/Unity-Technologies/ml-agents/issues/1786) and [here](https://answers.unity.com/questions/1528526/is-it-possible-to-run-a-game-built-in-unity-execut.html).
