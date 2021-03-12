import numpy as np
from psm.util import unit_ball_measure, is_on_manifold, Projection
from psm.manifolds import ManifoldStack
from psm.tasks import Task
from psm.algorithms.rrt_star_manifold import RRTStarManifold


class IKRRTStar:
    def __init__(self,
                 task: Task,
                 cfg: dict):
        self.name = "IK_RRT"
        self.n_manifolds = len(task.manifolds)
        self.task = task
        self.start = task.start
        self.cfg = cfg
        self.n_samples = cfg['N']
        self.alpha = cfg['ALPHA']
        self.beta = cfg['BETA']
        self.eps = cfg['EPS']
        self.rho = cfg['RHO']
        self.r_max = cfg['R_MAX']
        self.collision_res = cfg['COLLISION_RES']
        self.d = task.d

        self.lim_lo = task.lim_lo
        self.lim_up = task.lim_up
        self.gamma = np.power(2 * (1 + 1.0 / float(self.d)), 1.0 / float(self.d)) * \
                     np.power(task.get_joint_space_volume() / unit_ball_measure(self.d), 1. / float(self.d))

        self.Q_near_ids = []
        self.G = None
        self.G_list = []
        self.V_goal = []
        self.V_goal_list = []
        self.path = None
        self.path_id = None

        # check if start point is on first manifold
        if not is_on_manifold(task.manifolds[0], task.start, self.eps):
            raise Exception('The start point is not on the manifold h(start)= ' + str(task.manifolds[0].y(task.start)))

    def run(self) -> bool:
        # iterate over sequence of manifolds
        q_start = self.start.copy()
        cost = 0.0
        path = []
        for n in range(self.n_manifolds - 1):
            print('######################################################')
            print('n', n)
            print('Active Manifold: ', self.task.manifolds[n].name)
            print('Target Manifold: ', self.task.manifolds[n + 1].name)

            # sample a goal configuration with IK
            curr_manifold = self.task.manifolds[n]
            next_manifold = ManifoldStack(manifolds=[self.task.manifolds[n], self.task.manifolds[n + 1]])

            ik_proj = Projection(f=next_manifold.y, J=next_manifold.J)

            res_plan = False
            max_goals = 10
            iter_goals = 0
            while not res_plan:
                res_proj = False
                while not res_proj:
                    q_rand = self.task.sample()
                    res_proj, q_goal = ik_proj.project(q_rand)
                    if not self.task.is_valid_conf(q_goal):
                        res_proj = False
                    if self.task.is_collision_conf(q_goal):
                        res_proj = False

                # plan path to goal configuration with RRT*
                rrt_task = Task('empty')
                rrt_task.start = q_start
                rrt_task.goal = q_goal
                rrt_task.obstacles = self.task.obstacles
                planner = RRTStarManifold(rrt_task, curr_manifold, self.cfg)
                path_idx, opt_path = planner.run()
                if path_idx:
                    q_reached = planner.G.V[path_idx[-1]].value
                    res_plan = np.linalg.norm(q_reached - q_goal) < self.eps
                else:
                    res_plan = False

                if not res_plan:
                    iter_goals += 1

                if iter_goals == max_goals:
                    return False

            cost += planner.G.comp_opt_path(q_goal)
            path += [[planner.G.V[idx].value for idx in planner.G.path]]

            q_start = q_goal.copy()

            # store results for later evaluation
            self.G_list.append(planner.G)
            self.V_goal_list.append([0])

        self.path = path
        return True
