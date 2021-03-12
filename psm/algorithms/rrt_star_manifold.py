import tqdm
import numpy as np
from math import ceil
from psm.tree import Tree
from psm.util import unit_ball_measure, Projection
from psm.tasks import Task
from psm.manifolds import Manifold


class RRTStarManifold:
    def __init__(self,
                 task: Task,
                 manifold: Manifold,
                 cfg: dict):
        self.name = "RRT_Manifold"
        self.task = task
        self.start_value = task.start
        self.goal_value = task.goal
        self.manifold = manifold
        self.n_samples = cfg['N']
        self.alpha = cfg['ALPHA']
        self.beta = cfg['BETA']
        self.conv_tol = cfg['CONV_TOL']
        self.collision_res = cfg['COLLISION_RES']
        self.proj_step_size = cfg['PROJ_STEP_SIZE']
        self.d = task.d

        self.G = Tree(task.d, exact_nn=False)
        self.result = False
        self.Q_near_ids = []

        self.lim_lo = task.lim_lo
        self.lim_up = task.lim_up
        self.gamma = np.power(2 * (1 + 1.0 / float(self.d)), 1.0 / float(self.d)) * \
                     np.power(task.get_joint_space_volume() / unit_ball_measure(self.d), 1. / float(self.d))

    def steer(self,
              q_from: np.ndarray,
              q_to: np.ndarray) -> np.ndarray:
        if np.linalg.norm(q_to - q_from) < self.alpha:
            q_new = np.copy(q_to)
        else:
            diff = q_to - q_from
            q_new = q_from + self.alpha * diff * (1.0 / np.linalg.norm(diff))
        return q_new

    def is_collision(self,
                     q_a: np.ndarray,
                     q_b: np.ndarray) -> bool:
        N = int(ceil(np.linalg.norm(q_b - q_a) / self.collision_res))
        for i in range(N + 1):
            q = q_a if N == 0 else q_a + i / float(N) * (q_b - q_a)
            res = self.task.is_collision_conf(q)
            if res:
                return True
        return False

    def extend(self,
               q_from: np.ndarray,
               q_from_id: int,
               q_to: np.ndarray,
               q_to_id: int) -> bool:
        if not self.is_collision(q_from, q_to):
            n = float(len(self.G.V))
            r = min([self.gamma * np.power(np.log(n) / n, 1.0 / self.d), self.alpha])

            self.Q_near_ids = self.G.get_nearest_neighbors(node_value=q_to, radius=r)

            c_min = self.G.V[q_from_id].cost + np.linalg.norm(q_from - q_to)
            c_min_inc = np.linalg.norm(q_from - q_to)
            q_min_idx = q_from_id
            for idx in self.Q_near_ids:
                q_idx = self.G.V[idx].value
                c_idx = self.G.V[idx].cost + np.linalg.norm(q_idx - q_to)

                if not self.is_collision(q_idx, q_to) and c_idx < c_min:
                    c_min = c_idx
                    c_min_inc = np.linalg.norm(q_idx - q_to)
                    q_min_idx = idx

            self.G.add_node(node_id=q_to_id, node_value=q_to, node_cost=c_min, inc_cost=c_min_inc)
            self.G.add_edge(edge_id=self.G.edge_count, node_a=q_min_idx, node_b=q_to_id)

            if np.linalg.norm(q_to - self.goal_value) < self.conv_tol:
                self.result = True
            return True

        return False

    def rewire(self,
               q_from: np.ndarray,
               q_from_id: int):
        for idx in self.Q_near_ids:  # Q_near_ids was previously computed in extend function
            q_idx = self.G.V[idx].value
            c_idx = self.G.V[idx].cost
            c_new = self.G.V[q_from_id].cost + np.linalg.norm(q_from - q_idx)

            if not self.is_collision(q_from, q_idx) and c_new < c_idx:
                idx_parent = self.G.V[idx].parent
                self.G.remove_edge(idx_parent, idx)
                self.G.add_edge(edge_id=self.G.edge_count, node_a=q_from_id, node_b=idx)
                self.G.V[idx].cost = c_new
                self.G.V[idx].parent = q_from_id
                self.G.V[idx].inc_cost = np.linalg.norm(q_from - q_idx)
                self.G.update_child_costs(node_id=idx)

                # check for convergence
                if np.linalg.norm(q_idx - self.goal_value) < self.conv_tol:
                    self.result = True

    def run(self) -> (list, list):
        # the start node is only node in tree with id=0, cost=0, parent=None
        self.G.add_node(node_id=0, node_value=self.start_value, node_cost=0.0, inc_cost=0.0)
        self.result = False

        proj = Projection(f=self.manifold.y, J=self.manifold.J, step_size=self.proj_step_size)
        pbar = tqdm.tqdm(total=self.n_samples)
        for i in range(self.n_samples):
            pbar.update()
            if np.random.rand() < self.beta:
                q_target = self.goal_value
            else:
                q_target = self.task.sample()

            q_near, q_near_id = self.G.get_nearest_neighbor(node_value=q_target)
            q_new = self.steer(q_near, q_target)
            q_new_idx = self.G.node_count

            res, q_new = proj.project(q_new)
            if not res:
                continue

            extended = self.extend(q_from=q_near, q_from_id=q_near_id, q_to=q_new, q_to_id=q_new_idx)
            if extended:
                self.rewire(q_from=q_new, q_from_id=q_new_idx)

        pbar.close()
        print('')

        self.G.comp_opt_path(self.goal_value, self.conv_tol)
        opt_path = [self.G.V[idx].value for idx in self.G.path]
        return self.G.path, opt_path
