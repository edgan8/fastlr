import numpy as np
import ray
import scipy.optimize
import time
from multiprocessing import Pool
import os

ray.init(num_cpus=16)

@ray.remote
def task_async():
    time.sleep(.5)
    return 1

def task_sync(x):
    time.sleep(.5)
    return 1

def job_sync(num_tasks, n):
    task_results = [task_sync(n) for i in range(num_tasks)]
    return sum(task_results)

def job_async(num_tasks, n):
    task_result_ids = [
        task_async.remote()
        for task_id in range(num_tasks)
    ]
    task_results = ray.get(task_result_ids)
    ray.timeline(filename="timeline.json")
    return sum(task_results)

def job_pool(num_tasks, n):
    p = Pool(processes=8)
    res = p.imap_unordered(task_sync_hard, range(num_tasks))
    return sum(res)


def main():
    print("Starting")
    print("Nodes: {}".format(ray.nodes()))
    num_tasks = 100
    n = .5
    start_time = time.time()
#    job_sync(num_tasks, n=n)
    job_async(num_tasks, n=n)
#    job_pool(num_tasks, n=n)
    end_time = time.time()
    print("Elapsed: {}".format(end_time - start_time))

if __name__ == "__main__":
    main()
