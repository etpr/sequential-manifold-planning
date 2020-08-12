import numpy as np
from smp.smp import SMP
from smp.util import read_cfg
from smp.task import Task

parameters = \
            '[general]\n\
            SEED          = 1\n\
            CONV_TOL      = 0.1\n\
            N             = 2000\n\
            ALPHA         = 0.5\n\
            BETA          = 0.05\n\
            COLLISION_RES = 0.1\n\
            EPS           = 1e-2\n\
            RHO           = 1e-1\n\
            R_MAX         = 1.5'

task_name = 'hourglass'  # 'sphere' 'hourglass' 'hourglass_obstacles'

cfg = read_cfg(parameters)
np.random.seed(cfg['SEED'])
task = Task(task_name)
planner = SMP(task=task, cfg=cfg)
path_idx, path = planner.run()

if path_idx is not None:
    task.plot(planner.G_list, planner.V_goal_list, path)
