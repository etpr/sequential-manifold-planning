import numpy as np
from psm.tasks import Task
from psm.util import read_cfg
from psm.algorithms.psm import PSM
from psm.algorithms.psm_single_tree import PSMSingleTree
from psm.algorithms.ik_rrt_star import IKRRTStar
from psm.algorithms.cbirrt import CBIRRT
from psm.algorithms.random_mmp import RandomMMP

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

task_names = ['3d_point_w_obstacles', '3d_point_wo_obstacles']
for task_name in task_names:
    cfg = read_cfg(parameters)
    np.random.seed(cfg['SEED'])
    task = Task(task_name)

    planner = PSM(task=task, cfg=cfg)
    if planner.run():
        task.plot(planner.name, planner.G_list, planner.V_goal_list, planner.path)

    planner = PSMSingleTree(task=task, cfg=cfg)
    if planner.run():
        task.plot(planner.name, [planner.G], None, planner.path)

    planner = IKRRTStar(task=task, cfg=cfg)
    if planner.run():
        task.plot(planner.name, planner.G_list, planner.V_goal_list, planner.path)

    planner = CBIRRT(task=task, cfg=cfg)
    if planner.run():
        task.plot(planner.name, [planner.G_a, planner.G_b], [], planner.path)

    planner = RandomMMP(task=task, cfg=cfg)
    if planner.run():
        task.plot(planner.name, None, None, planner.path)
