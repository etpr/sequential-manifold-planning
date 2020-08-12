import numpy as np
from scipy.linalg import null_space
from math import ceil
from smp.tree import Tree
from smp.util import Projection, unit_ball_measure, is_on_manifold
from smp.features import FeatureStack


class SMP:
    def __init__(self, task, cfg):
        self.n_manifolds = len(task.manifolds)
        self.task = task
        self.start = task.start
        self.n_samples = cfg['N']
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
                     np.power(self.task.getJointSpaceVolume() / unit_ball_measure(self.d), 1. / float(self.d))

        self.Q_near_ids = []
        self.G = None
        self.G_list = []
        self.V_goal = []
        self.V_goal_list = []

        # check if start point is on first manifold
        if not is_on_manifold(task.manifolds[0], task.start, self.eps):
            raise Exception('The start point is not on the manifold h(start)= ' + str(task.manifolds[0].y(task.start)))

        # check if start point is in collision
        if self.task.is_collision_conf(task.start):
            raise Exception('The start point is in collision.')

    def run(self):
        # iterate over sequence of manifolds
        for n in range(self.n_manifolds - 1):
            self.G = Tree(self.d, exact_nn=False)
            if n == 0:
                # init tree with start node
                self.G.add_node(node_id=0, node_value=self.start, node_cost=0.0, inc_cost=0.0)
            else:
                # init tree with transition nodes
                self.G.add_node(node_id=0, node_value=np.ones(self.d) * np.inf, node_cost=0.0, inc_cost=0.0)  # virtual root node
                for idx, v_id in enumerate(self.V_goal):
                    node_id = self.G_list[-1].V[v_id]
                    self.G.add_node(node_id=idx+1, node_value=node_id.value, node_cost=node_id.cost, inc_cost=node_id.cost)
                    self.G.add_edge(edge_id=self.G.edge_count, node_a=0, node_b=idx+1)

            self.V_goal.clear()

            self.grow_tree(curr_manifold=self.task.manifolds[n],
                           next_manifold=self.task.manifolds[n+1])

            # only continue with best intersection node
            if self.greedy:
                idx = min(self.V_goal, key=lambda idx: np.linalg.norm(self.G.V[idx].cost))
                self.V_goal = [idx]

            # store results for later evaluation
            self.G_list.append(self.G)
            self.V_goal_list.append(self.V_goal.copy())

            if len(self.V_goal) == 0:
                print('RRT extension did not reach any intersection nodes')
                return None, None
            else:
                print('number of nodes in tree_' + str(n) + ' = ' + str(len(self.G.V)))
                print('number of goal nodes in tree_' + str(n) + ' = ' + str(len(self.V_goal)))

        # compute optimal path
        opt_idx = min(self.V_goal, key=lambda idx: np.linalg.norm(self.G.V[idx].cost))
        path_idx = self.G.comp_path(opt_idx)
        opt_path = [[self.G.V[idx].value for idx in path_idx]]
        opt_path_idx = [path_idx.copy()]
        opt_path_cost = self.G.V[opt_idx].cost

        for G, V_goal in zip(reversed(self.G_list[:-1]), reversed(self.V_goal_list[:-1])):
            opt_idx = V_goal[path_idx[0] - 1]  # -1 offset due to virtual root node
            path_idx = G.comp_path(opt_idx)
            opt_path_idx.append(path_idx.copy())
            opt_path.append([G.V[idx].value for idx in path_idx])

        print('number of nodes in optimal path = ', str(sum([len(p) for p in opt_path_idx])))
        print('cost of optimal path = ', "{:.2f}".format(opt_path_cost))

        return list(reversed(opt_path_idx)), list(reversed(opt_path))

    def sample(self):
        q_target = np.empty(self.d)
        for i in range(self.d):
            q_target[i] = self.lim_lo[i] + np.random.rand() * (self.lim_up[i] - self.lim_lo[i])

        return q_target

    def steer(self, q_from, q_from_id, q_to, curr_manifold, next_manifold):
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

    def is_collision(self, q_a, q_b):
        N = int(ceil(np.linalg.norm(q_b - q_a) / self.collision_res))
        for i in range(N + 1):
            q = q_a if N == 0 else q_a + i / float(N) * (q_b - q_a)
            res = self.task.is_collision_conf(q)
            if res:
                return True
        return False

    def extend(self, q_from, q_from_id, q_to, q_to_idx):
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

        self.G.add_node(node_id=q_to_idx, node_value=q_to, node_cost=c_min, inc_cost=c_min_inc)
        self.G.add_edge(edge_id=self.G.edge_count, node_a=q_min_idx, node_b=q_to_idx)

        return True

    def rewire(self, q_from, q_from_id):
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

    def grow_tree(self, curr_manifold, next_manifold):
        curr_projector = Projection(f_=curr_manifold.y, J_=curr_manifold.J, step_size_=self.proj_step_size)
        joint_manifold = FeatureStack([curr_manifold, next_manifold])
        joint_projector = Projection(f_=joint_manifold.y, J_=joint_manifold.J, step_size_=self.proj_step_size)

        for i in range(self.n_samples):
            q_target = self.sample()
            q_near, q_near_id = self.G.get_nearest_neighbor(node_value=q_target)

            q_new = self.steer(q_near, q_near_id, q_target, curr_manifold, next_manifold)

            if q_new is None:
                continue

            # project q_new on current or next manifold
            on_next_manifold = False
            if np.linalg.norm(next_manifold.y(q_new)) < self.r_max:  # np.random.rand() * self.r_max:
                res, q_new_proj = joint_projector.project(q_new)
                if np.linalg.norm(q_new_proj - q_near) > self.alpha:
                    res, q_new_proj = curr_projector.project(q_new)
            else:
                res, q_new_proj = curr_projector.project(q_new)

            if not res:
                continue  # continue if projection was not successful

            # check if q_new_proj is on the next manifold
            if is_on_manifold(next_manifold, q_new_proj, self.eps):
                if len(self.V_goal) == 0:
                    on_next_manifold = True
                else:
                    q_proj_near = min(self.V_goal, key=lambda idx: np.linalg.norm(self.G.V[idx].value - q_new_proj))
                    if np.linalg.norm(self.G.V[q_proj_near].value - q_new_proj) > self.rho:
                        on_next_manifold = True
                    else:
                        continue  # continue if a node close to q_new_proj is already in the tree

            q_new_idx = self.G.node_count
            extended = self.extend(q_from=q_near, q_from_id=q_near_id, q_to=q_new_proj, q_to_idx=q_new_idx)

            if extended:
                self.rewire(q_from=q_new_proj, q_from_id=q_new_idx)

                # add to V_goal if q_new is on the next manifold
                if on_next_manifold:
                    self.V_goal.append(q_new_idx)
