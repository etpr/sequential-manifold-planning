import numpy as np
from scipy.linalg import null_space
from math import ceil
import tqdm
from psm.tasks import Task
from psm.tree import Tree
from psm.util import unit_ball_measure, is_on_manifold, get_volume, Projection
from psm.manifolds import ManifoldStack, Manifold


class PSMSingleTree:
    def __init__(self,
                 task: Task,
                 cfg: dict):
        self.name = "PSM_Single_Tree"
        self.n_manifolds = len(task.manifolds)
        self.task = task
        self.start = task.start
        self.n_samples = cfg['N'] * (self.n_manifolds - 1)
        self.alpha = cfg['ALPHA']
        self.beta = cfg['BETA']
        self.eps = cfg['EPS']
        self.rho = cfg['RHO']
        self.r_max = cfg['R_MAX']
        self.collision_res = cfg['COLLISION_RES']
        self.d = task.d
        self.greedy = cfg['GREEDY']
        self.proj_step_size = cfg['PROJ_STEP_SIZE']

        self.lim_lo = task.lim_lo
        self.lim_up = task.lim_up
        self.gamma = np.power(2 * (1 + 1.0 / float(self.d)), 1.0 / float(self.d)) * \
                     np.power(get_volume(self.lim_lo, self.lim_up) / unit_ball_measure(self.d), 1. / float(self.d))

        self.Q_near_ids = []
        self.G = None
        self.V_goals = []
        self.path = None
        self.path_id = None

        # check if start point is on first manifold
        if not is_on_manifold(task.manifolds[0], task.start, self.eps):
            raise Exception('The start point is not on the manifold h(start)= ' + str(task.manifolds[0].y(task.start)))

        # check if start point is in configuration space limits
        if not self.task.is_valid_conf(task.start):
            raise Exception('The start point is not in the system limits.')

        # check if start point is in collision
        if self.task.is_collision_conf(task.start):
            raise Exception('The start point is in collision.')

    def run(self) -> bool:
        curr_projectors = []
        next_projectors = []
        curr_manifolds = []
        next_manifolds = []
        # iterate over sequence of manifolds
        for n in range(self.n_manifolds - 1):
            print('######################################################')
            print('n', n)
            print('Active Manifold: ', self.task.manifolds[n].name)
            print('Target Manifold: ', self.task.manifolds[n+1].name)

            curr_manifold = self.task.manifolds[n]
            next_manifold = self.task.manifolds[n + 1]
            joint_manifold = ManifoldStack([curr_manifold, next_manifold])

            # initiate manifold and projector sequence
            curr_manifolds += [curr_manifold]
            next_manifolds += [next_manifold]
            curr_projectors += [Projection(f=curr_manifold.y, J=curr_manifold.J, step_size=self.proj_step_size)]
            next_projectors += [Projection(f=joint_manifold.y, J=joint_manifold.J, step_size=self.proj_step_size)]

            if n < self.n_manifolds - 1:
                self.V_goals += [[]]

        self.G = Tree(self.d, exact_nn=False)
        self.G.add_node(node_id=0, node_value=self.start, node_cost=0.0, inc_cost=0.0)
        self.G.V[0].aux = 0  # store manifold id for every node

        pbar = tqdm.tqdm(total=self.n_samples)
        for i in range(self.n_samples):
            pbar.update()
            q_target = self.task.sample()
            q_near, q_near_id = self.G.get_nearest_neighbor(node_value=q_target)
            manifold_id = self.G.V[q_near_id].aux

            if manifold_id >= self.n_manifolds - 1:
                continue  # do not extend goal nodes

            if is_on_manifold(self.task.manifolds[manifold_id + 1], q_near):
                # move node directly to next manifold
                self.G.V[q_near_id].aux += 1
                continue

            curr_manifold = curr_manifolds[manifold_id]
            next_manifold = next_manifolds[manifold_id]
            curr_projector = curr_projectors[manifold_id]
            joint_projector = next_projectors[manifold_id]
            q_new = self.steer(q_near,
                               q_near_id,
                               q_target,
                               curr_manifold,
                               next_manifold)

            if q_new is None:
                continue

            # project q_new on current or next manifold
            on_next_manifold = False
            if np.linalg.norm(next_manifold.y(q_new)) < np.random.rand() * self.r_max:
                res, q_new_proj = joint_projector.project(q_new)
            else:
                res, q_new_proj = curr_projector.project(q_new)

            if not res:
                continue  # continue if projection was unsuccessful

            # check if q_new_proj is on the next manifold
            if is_on_manifold(next_manifold, q_new_proj, self.eps):
                on_next_manifold = True
                if len(self.V_goals[manifold_id]) > 0:
                    q_proj_near = min(self.V_goals[manifold_id], key=lambda idx: np.linalg.norm(self.G.V[idx].value - q_new_proj))
                    if np.linalg.norm(self.G.V[q_proj_near].value - q_new_proj) < self.rho:
                        continue  # continue if a node close to q_new_proj is already in the tree

            q_new_idx = self.G.node_count
            extended = self.extend(q_from=q_near,
                                   q_from_id=q_near_id,
                                   q_to=q_new_proj,
                                   q_to_id=q_new_idx,
                                   manifold_id=manifold_id + on_next_manifold)

            if extended:
                self.rewire(q_from=q_new_proj, q_from_id=q_new_idx)

                if on_next_manifold:
                    # add to V_goal if q_new is on the next manifold
                    self.V_goals[manifold_id].append(q_new_idx)

        pbar.close()
        print('')
        if len(self.V_goals[-1]) == 0:
            return False

        opt_idx = min(self.V_goals[-1], key=lambda idx: np.linalg.norm(self.G.V[idx].cost))
        opt_path_idx = self.G.comp_path(opt_idx)

        # split into individual path segments per manifold
        opt_path = []
        opt_path_m = []
        m = 0
        for idx in opt_path_idx:
            opt_path_m += [self.G.V[idx].value]
            if self.G.V[idx].aux != m:
                opt_path += [opt_path_m.copy()]
                opt_path_m = [self.G.V[idx].value]
                m += 1

        if len(opt_path) != self.n_manifolds - 1:
            return False

        self.path = opt_path
        self.path_idx = opt_path_idx
        return True

    def steer(self,
              q_from: np.ndarray,
              q_from_id: int,
              q_to: np.ndarray,
              curr_manifold: Manifold,
              next_manifold: Manifold) -> np.ndarray:
        if np.random.rand() < self.beta and not self.G.V[q_from_id].con_extend:
            # steer towards next_manifolds
            self.G.V[q_from_id].con_extend = True
            yn = next_manifold.y(q_from)
            Jn = next_manifold.J(q_from)

            d = -Jn.T @ yn
            # project on current manifold
            J = null_space(curr_manifold.J(q_from))
            if J.shape[1] != 0:
                d = J @ J.T @ d

        else:
            # steer towards q_to
            d = (q_to - q_from)

            # project on current manifold
            J = null_space(curr_manifold.J(q_from))
            if J.shape[1] != 0:
                d = J @ J.T @ d

        if np.linalg.norm(d) > 0.0:
            q_new = q_from + self.alpha * d * (1.0 / np.linalg.norm(d))
            return q_new
        else:
            return None

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
               q_to_id: int,
               manifold_id: int) -> bool:
        if self.is_collision(q_from, q_to):
            return False

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
        self.G.V[q_to_id].aux = manifold_id
        self.G.add_edge(edge_id=self.G.edge_count, node_a=q_min_idx, node_b=q_to_id)

        return True

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
