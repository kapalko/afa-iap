import os
import sys
import importlib
import socket
import random
import gym
import airforce_arcade

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


def register_custom_model(model_config):
    if "custom_model" not in model_config.keys():
        return

    custom_model_name = model_config["custom_model"]
    model_module = importlib.import_module('.'+custom_model_name, "model")
    ModelCatalog.register_custom_model(custom_model_name, model_module.CustomModel)


def register_custom_env(env_name, env_config):
    # NOTE: even though airforce-arcade envs are already registered in gym,
    #       we need to re-register and overwrite the original one to specify
    #       environment configuration.
    if env_name not in airforce_arcade.envs_list:
        return

    def _env(config):
        import airforce_arcade # need this import for remote ray process
        if not config.worker_index:
            config.worker_index = 0
        # Randomly get unused port since some ports closed to the base port (5005) may be used
        port_range = env_config.pop("port_range") if "port_range" in env_config.keys() else [5005, 6000]
        port_not_in_use = [port for port in range(*port_range) if not is_port_in_use(port)]
        worker_id = port_not_in_use[config.worker_index] - 5005 # convert port to worker_id (only determine port in env creation)
        return gym.make(env_name, worker_id=worker_id, **env_config)
    register_env(env_name, _env)


def is_port_in_use(port):
    if sys.platform in ['linux', 'linux2']:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("localhost", port))
        except:
            return True
        finally:
            s.close()
            return False
    elif sys.platform == 'win32':
        return False # TODO: check port use in windows
    else:
        raise NotImplementedError("Check-used-port function is not implemented in {} system.".format(sys.platform))