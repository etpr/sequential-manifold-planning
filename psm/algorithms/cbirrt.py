import numpy as np
from math import ceil
from psm.tree import Tree
from psm.util import is_on_manifold, path_cost, Projection
from psm.manifolds import ManifoldStack
from psm.tasks import Task


class CBIRRT:
    def __init__(self,
                 task: Task,
                 cfg: dict):
        self.name = "CBIRRT"
        self.n_manifolds = len(task.manifolds)
        self.task = task
        self.start = task.start
        self.n_samples = cfg['N']
        self.alpha = cfg['ALPHA']
        self.eps = cfg['EPS']
        self.collision_res = cfg['COLLISION_RES']
        self.d = task.d

        self.lim_lo = task.lim_lo
        self.lim_up = task.lim_up
        self.result = False
        self.goal = task.goal
        self.path = None
        self.path_id = None

        self.G_a = Tree(self.d, exact_nn=False)
        self.G_b = Tree(self.d, exact_nn=False)

        self.manifold_projectors = []
        for m in self.task.manifolds:
            self.manifold_projectors.append(Projection(f=m.y, J=m.J))

        # check if start point is on first manifold
        if not is_on_manifold(task.manifolds[0], task.start, self.eps):
            raise Exception('The start point is not on the manifold h(start)= ' + str(task.manifolds[0].y(task.start)))

        # check if start point is in collision
        if self.task.is_collision_conf(task.start):
            raise Exception('The start point is in collision.')

    def run(self) -> bool:
        self.G_a.add_node(node_id=0, node_value=self.start, node_cost=0.0, inc_cost=0.0)
        self.G_b.add_node(node_id=0, node_value=self.goal, node_cost=0.0, inc_cost=0.0)
        self.G_a.V[0].aux = 0
        self.G_b.V[0].aux = self.n_manifolds - 1

        # grow tree with bidirectional strategy
        for i in range(self.n_samples):
            q_rand = self.task.sample()
            q_near_a, q_near_a_id = self.G_a.get_nearest_neighbor(node_value=q_rand)
            q_reached_a = self.constrained_extend(self.G_a, q_near_a, q_near_a_id, q_rand)
            q_near_b, q_near_b_id = self.G_b.get_nearest_neighbor(node_value=q_reached_a)
            q_reached_b = self.constrained_extend(self.G_b, q_near_b, q_near_b_id, q_reached_a)
            if np.isclose(q_reached_a, q_reached_b).all():
                cost_a = self.G_a.comp_opt_path(q_reached_a)
                cost_b = self.G_b.comp_opt_path(q_reached_b)
                path_idx = [self.G_a.path, list(reversed(self.G_b.path))]
                path = [self.G_a.V[idx].value for idx in self.G_a.path] + \
                       [self.G_b.V[idx].value for idx in list(reversed(self.G_b.path))]
                path_m = [self.G_a.V[idx].aux for idx in self.G_a.path] + \
                         [self.G_b.V[idx].aux for idx in list(reversed(self.G_b.path))]

                if not np.isclose(path[0], self.start).all():
                    path_idx.reverse()
                    path.reverse()
                    path_m.reverse()

                path_sc = self.shortcut(path, N=2000)

                # split the found path into subpath corresponding to the individual manifolds
                path_sc_out = []
                i_m = 1
                manifold_idx = [0]
                for idx, q in enumerate(path_sc):
                    if is_on_manifold(self.task.manifolds[i_m], q):
                        manifold_idx += [idx]
                        i_m += 1
                        if i_m == len(self.task.manifolds):
                            break

                for i in range(len(manifold_idx) - 1):
                    path_sc_out += [path_sc[manifold_idx[i]:manifold_idx[i+1] + 1]]

                i_m = 1
                manifold_idx = [0]
                path_out = []
                path_idx_out = []
                for idx, q in enumerate(path):
                    if is_on_manifold(self.task.manifolds[i_m], q):
                        manifold_idx += [idx]
                        i_m += 1
                        if i_m == len(self.task.manifolds):
                            break

                for i in range(len(manifold_idx) - 1):
                    path_out += [path[manifold_idx[i]:manifold_idx[i + 1] + 1]]
                    path_idx_out += [path_idx[manifold_idx[i]:manifold_idx[i + 1] + 1]]

                self.path_id = path_idx_out
                self.path = path_sc_out
                return True
            else:
                self.G_a, self.G_b = self.G_b, self.G_a

        return False

    def shortcut(self,
                 path: list,
                 N: int = 100) -> list:
        path_sc = path.copy()
        for n in range(N):
            m = len(path_sc)
            i = np.random.randint(0, m - 1)
            j = np.random.randint(i, m)
            q_i = path_sc[i]
            q_j = path_sc[j]
            G = Tree(self.d, exact_nn=False)
            G.add_node(0, q_i, 0, 0)
            h = [np.linalg.norm(m.y(q_i)) for m in self.task.manifolds]
            G.V[0].aux = np.argmin(h)
            q_reach = self.constrained_extend(G, q_i, 0, q_j)

            if np.linalg.norm(q_reach - q_j) < self.eps:
                if G.comp_opt_path(q_reach) < path_cost(path_sc[i:j+1]):
                    path_i_j = [G.V[idx].value for idx in G.path]
                    path_sc = path_sc[:i] + path_i_j + path_sc[j+1:]

        return path_sc

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

    def constrained_extend(self,
                           G: Tree,
                           q_from: np.ndarray,
                           q_from_id: int,
                           q_to: np.ndarray) -> np.ndarray:
        q_s = q_from.copy()
        q_s_id = q_from_id
        q_s_old = q_s.copy()

        while True:
            if np.isclose(q_s, q_to).all():
                return q_s
            if np.linalg.norm(q_s - q_to) > np.linalg.norm(q_s_old - q_to):
                return q_s_old

            q_s_old = q_s.copy()
            q_s_old_id = q_s_id

            # step towards the target configuration
            q_s = self.steer(q_from=q_s, q_to=q_to)

            # project q_s onto nearest manifold
            h = [np.linalg.norm(m.y(q_s)) for m in self.task.manifolds]
            idx = np.argmin(h)

            if idx != G.V[q_s_id].aux:
                curr_manifold = self.task.manifolds[G.V[q_s_id].aux]
                next_manifold = self.task.manifolds[idx]
                joint_manifold = ManifoldStack([curr_manifold, next_manifold])
                joint_projector = Projection(joint_manifold.y, joint_manifold.J)
                res, q_s = joint_projector.project(q_s)
            else:
                res, q_s = self.manifold_projectors[idx].project(q_s)

            if res and not self.is_collision(q_s_old, q_s):
                inc_cost = np.linalg.norm(q_s - q_s_old)
                node_cost = G.V[q_s_old_id].cost + inc_cost
                q_s_id = G.node_count
                G.add_node(node_id=q_s_id, node_value=q_s, node_cost=node_cost, inc_cost=inc_cost)
                G.add_edge(edge_id=G.edge_count, node_a=q_s_old_id, node_b=q_s_id)
                G.V[q_s_id].aux = idx
                if np.isclose(q_s, q_s_old).all():
                    return q_s
            else:
                return q_s_old
