import numpy as np
import ray
import scipy.optimize
import time

ray.init()

@ray.remote
def task_async(n):
    return sum([i**2 for i in range(n)])

def task_sync(n):
    return sum([i**2 for i in range(n)])

def job_sync(num_tasks, n):
    for task_id in range(num_tasks):
        task_sync(n)

def job_async(num_tasks, n):
    task_result_ids = [
        task_async.remote(n)
        for task_id in range(num_tasks)
    ]
    task_results = ray.get(task_result_ids)
    return sum(task_results)


def main():
    print("Starting")
    num_tasks = 100
    n = 1000000
    start_time = time.time()
    job_sync(num_tasks, n=n)
#    job_async(num_tasks, n=n)
    end_time = time.time()
    print("Elapsed: {}".format(end_time - start_time))

if __name__ == "__main__":
    main()
