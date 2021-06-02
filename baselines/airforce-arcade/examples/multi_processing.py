import os
import random
import time
import subprocess
import numpy
from functools import partial
import ray
import multiprocessing as mp


def main():
    n_workers = 7
    with_ray(n_workers, CreateEnv)


@ray.remote
class CreateEnv:
    def run(self, id):
        # we can run 10-12 workers if not using circle abstraction and 7 workers with it.
        import gym
        import airforce_arcade
        port_not_in_use = [port for port in range(5005, 6000) if not is_port_in_use(port)]
        self.id = id
        id = random.choice(port_not_in_use) - 5005 # 5005 is the base port of unity environment class
        self.env = gym.make("TestingRoomICRA-v0", worker_id=id, allow_multiple_obs=True, use_circle_abstraction=True, uint8_visual=True)
        _ = self.env.step(self.env.action_space.sample())
        return "[Worker {}] Successfully create and step environment.".format(id)
    
    def close(self):
        self.env.close()
        print("[Worker {}] Closed!".format(self.id))


@ray.remote
class OpenUnityStandalone:
    def run(self, id):
        from mlagents_envs import env_utils
        dirname = os.path.dirname(__file__)
        fpath = os.path.join(dirname, "../builds/TestingRoomICRA/Linux/TestingRoomICRA_Linux.x86_64")
        self.proc = env_utils.launch_executable(fpath, [])
        self.id = id
        time.sleep(5)
        return "[Worker {}] Successfully open an unity standalone.".format(id)

    def close(self):
        self.proc.kill()
        print("[Worker {}] Closed!".format(self.id))


def with_ray(n_workers, actor_cls):
    ray.init()

    obj_refs = []
    actors = []
    for i in range(n_workers):
        actor = actor_cls.remote()
        obj_refs.append(actor.run.remote(i))
        actors.append(actor)

    timeout_cnt = 120
    curr_cnt = 0
    wait_refs = obj_refs
    n_complete = 0
    while len(wait_refs) != 0 and curr_cnt < timeout_cnt:
        ready_refs, wait_refs = ray.wait(wait_refs, timeout=1) # return one ready job at a time
        time.sleep(1)
        curr_cnt += 1
        if len(ready_refs) > 0:
            n_complete += 1
            print(ray.get(ready_refs[0]))
        print("[Time {:03d}] {}/{} workers running.".format(curr_cnt, n_complete, len(obj_refs)))


    for actor in actors:
        actor.close.remote() # close job


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == "__main__":
    main()