# run using: python enjoy.py folder_of_RLLib_model_output_folder_that_contains_params.json checkpt_num

import os
import numpy as np
import cv2
import tensorflow as tf
import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import sys
import csv
import datetime
import json
from airforce_arcade.envs import CanyonRun, DroneDodgeBall, Refueling, TimedWaypoints
tf.compat.v1.disable_v2_behavior()

# set these!
NUM_GAMES = 100
NUM_GPU = 0
NUM_WORKERS = 0
MODEL_DIR = sys.argv[1]
CHECKPT_NUM = sys.argv[2]
CHECKPT = MODEL_DIR + "/checkpoint_"+str(CHECKPT_NUM)+"/checkpoint-"+str(CHECKPT_NUM)

#allows to run on Supercloud when someone else owns /tmp/ray
TMP_DIR = '/tmp/guardianray/'

with open(MODEL_DIR+'/params.json') as f:
    paramfile = json.load(f)

IS_RANDOM = False

if len(sys.argv) == 4:
    IS_RANDOM = sys.argv[3] #optional argument

if IS_RANDOM == "random":
    random_act = True
else:
    random_act = False

envname = paramfile['env'][:paramfile['env'].find("-v0")]

OUTFILE = datetime.datetime.now().strftime(envname+"-testlog-%Y-%m-%d-%H-%M-%S-%f.csv")
FOUT = open(OUTFILE,"w")
writer = csv.writer(FOUT)
writer.writerow(["Episode Reward","Episode Length"])

paramfile['env_config']['env_build'] = None
#visualize environment
paramfile['env_config']['no_graphics'] = False
#set num workers to NUM_WORKERS
paramfile['num_workers'] = NUM_WORKERS
paramfile['num_gpus'] = NUM_GPU
envclass = globals()[envname]

def _env(config):
    return envclass(**paramfile['env_config'])

register_env(paramfile['env'], _env)

ray.init(num_gpus=NUM_GPU, _temp_dir = TMP_DIR)
agent = ppo.PPOTrainer(config=paramfile, env=paramfile['env'])
agent.restore(CHECKPT)
env = agent.workers.local_worker().env
if not random_act:
    for game in range(NUM_GAMES):
        num_steps = 0
        episode_rew = 0
        #action_hist = []
        done = False
        obs = env.reset()
        action = env.action_space.sample() #take random first action
        prev_act = env.action_space.sample()
        prev_rew = 0
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        state = state_init['default_policy']

        while not done:
            num_steps += 1
            env.render()
            action = agent.compute_action(obs, state, prev_action=prev_act, prev_reward=prev_rew, policy_id='default_policy')
            obs, reward, done, info = env.step(action)
            #action_hist.append(action)
            prev_rew = reward
            prev_act = action
            episode_rew += reward

        writer.writerow([episode_rew, num_steps])
        print("Game:", game)
        print("Episode Reward",episode_rew)
        print("Number of Steps",num_steps)
else: #taking random actions
    for game in range(NUM_GAMES):
        num_steps = 0
        episode_rew = 0
        #action_hist = []
        done = False
        obs = env.reset()
        action = env.action_space.sample() #take random first action

        while not done:
            num_steps += 1
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #action_hist.append(action)
            prev_rew = reward
            prev_act = action
            episode_rew += reward

        writer.writerow([episode_rew, num_steps])
        print("Game:", game)
        print("Episode Reward",episode_rew)
        print("Number of Steps",num_steps)

FOUT.close()
env.close()

