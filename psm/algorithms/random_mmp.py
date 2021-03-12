import numpy as np
from psm.tree import Tree
from psm.util import unit_ball_measure, is_on_manifold, path_cost, Projection
from psm.tasks import Task
from psm.algorithms.rrt_star_manifold import RRTStarManifold
from psm.manifolds import ManifoldStack


class RandomMMP:
    def __init__(self,
                 task: Task,
                 cfg: dict):
        self.name = 'Random_MMP'
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
        self.greedy = cfg['GREEDY']

        self.lim_lo = task.lim_lo
        self.lim_up = task.lim_up
        self.gamma = np.power(2 * (1 + 1.0 / float(self.d)), 1.0 / float(self.d)) * \
                     np.power(task.get_joint_space_volume() / unit_ball_measure(self.d), 1. / float(self.d))

        self.Q_near_ids = []
        self.G = None
        self.path = None
        self.path_id = None

        # check if start point is on first manifold
        if not is_on_manifold(task.manifolds[0], task.start, self.eps):
            raise Exception('The start point is not on the manifold h(start)= ' + str(task.manifolds[0].y(task.start)))

    def run(self) -> bool:
        # iterate over sequence of manifolds
        self.G = Tree(self.d, exact_nn=False)
        self.G.add_node(node_id=0, node_value=self.start, node_cost=0.0, inc_cost=0.0)
        self.G.V[0].aux = 0  # store manifold id for every node
        self.G.V[0].path = []

        for n in range(100):
            node_id = self.G.sample_node()
            curr_manifold_id = self.G.V[node_id].aux
            next_manifold_id = curr_manifold_id + 1
            q_start = self.G.V[node_id].value

            if is_on_manifold(self.task.manifolds[next_manifold_id], q_start):
                # move node directly to next manifold
                self.G.V[node_id].aux += 1
                q_reached = q_start
            else:
                # sample a goal configuration with IK
                curr_manifold = self.task.manifolds[curr_manifold_id]
                next_manifold = ManifoldStack(manifolds=[self.task.manifolds[curr_manifold_id],
                                                         self.task.manifolds[next_manifold_id]])

                ik_proj = Projection(f=next_manifold.y, J=next_manifold.J)
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
                planner = RRTStarManifold(task=rrt_task, manifold=curr_manifold, cfg=self.cfg)
                path_idx, opt_path = planner.run()

                result = False
                if path_idx:
                    q_reached = planner.G.V[path_idx[-1]].value
                    if np.linalg.norm(q_reached - q_goal) < self.eps:
                        result = True

                if not result:
                    continue

                cost = path_cost(opt_path)
                node_id_new = self.G.node_count
                self.G.add_node(node_id=node_id_new,
                                node_value=q_reached,
                                node_cost=self.G.V[node_id].cost + cost,
                                inc_cost=cost)
                self.G.V[node_id_new].aux = next_manifold_id
                self.G.V[node_id_new].path = opt_path
                self.G.add_edge(edge_id=self.G.edge_count,
                                node_a=node_id,
                                node_b=node_id_new)

            if next_manifold_id == self.n_manifolds - 1:
                self.G.comp_opt_path(q_reached, self.eps)
                opt_path = [self.G.V[idx].path for idx in self.G.path[1:]]
                self.path = opt_path
                return True

        return False

