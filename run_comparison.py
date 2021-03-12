import numpy as np
import timeit
from psm.tasks import Task
from psm.util import read_cfg, path_cost
from psm.algorithms.psm import PSM
from psm.algorithms.psm_greedy import PSMGreedy
from psm.algorithms.psm_single_tree import PSMSingleTree
from psm.algorithms.ik_rrt_star import IKRRTStar
from psm.algorithms.cbirrt import CBIRRT
from psm.algorithms.random_mmp import RandomMMP
from tabulate import tabulate

parameters = \
    '[general]\n\
    SEED          = 1\n\
    CONV_TOL      = 0.1\n\
    N             = 1200\n\
    ALPHA         = 1.0\n\
    BETA          = 0.1\n\
    COLLISION_RES = 0.1\n\
    EPS           = 1e-2\n\
    RHO           = 1e-1\n\
    R_MAX         = 1.5'

n_seed = 10
task_names = ['3d_point_wo_obstacles', '3d_point_w_obstacles']

for task_name in task_names:
    task = Task(task_name)
    cfg = read_cfg(parameters)
    seed_list = list(range(1, n_seed + 1))
    planner_list = [PSM, PSMSingleTree, PSMGreedy, IKRRTStar, CBIRRT, RandomMMP]
    results = [[task_name, 'Success', 'Path length', 'Comp. time']]

    for planner in planner_list:
        n_success = 0
        comp_time = []
        cost = []
        for seed in seed_list:
            cfg['SEED'] = seed
            np.random.seed(cfg['SEED'])
            p = planner(task=task, cfg=cfg)
            t = timeit.default_timer()
            if p.run():
                n_success += 1
                comp_time += [timeit.default_timer() - t]
                cost += [path_cost(p.path)]

        results += [[planner.__name__,
                     str(n_success) + ' / ' + str(n_seed),
                     "{:.2f}".format(np.mean(cost)) + "\\pm {:.2f}".format(np.std(cost)),
                     "{:.2f}".format(np.mean(comp_time)) + "\\pm {:.2f}".format(np.std(comp_time))]]

        print(tabulate(results))
