import numpy as np
from psm.manifolds import PointManifold, CylinderManifold, ParaboloidManifold
import plotly.graph_objs as go
from plotly.offline import plot
from psm.util import plot_box, check_limits


class Task:
    """This class represents a sequential motion planning task."""
    def __init__(self, name: str):
        self.name = name
        self.d = 3
        self.start = np.zeros(self.d)
        self.lim_lo = np.array([-6., -6., -6.])
        self.lim_up = np.array([6., 6., 6.])
        self.manifolds = []
        self.obstacles = []

        if name == 'empty':
            pass

        elif name == '3d_point_wo_obstacles':
            if self.d != 3:
                raise Exception('3d_point_wo_obstacles task only works with an ambient space dimensionality of 3')

            self.start = np.array([3.5, 3.5, 4.45])
            self.goal = np.array([-3.5, -3.5, -4.45])
            A = np.eye(2) * 0.1
            b = np.zeros(2)
            self.manifolds.append(ParaboloidManifold(A=A, b=b, c=2.0))
            self.manifolds.append(CylinderManifold(a=2.0, b=2.0))
            self.manifolds.append(ParaboloidManifold(A=-A, b=b, c=-2.0))
            self.manifolds.append(PointManifold(goal=self.goal))

        elif name == '3d_point_w_obstacles':
            if self.d != 3:
                raise Exception('3d_point_wo_obstacles task only works with an ambient space dimensionality of 3')

            self.start = np.array([3.5, 3.5, 4.45])
            self.goal = np.array([-3.5, -3.5, -4.45])
            A = np.eye(2) * 0.1
            b = np.zeros(2)
            self.manifolds.append(ParaboloidManifold(A=A, b=b, c=2.0))
            self.manifolds.append(CylinderManifold(a=2.0, b=2.0))
            self.manifolds.append(ParaboloidManifold(A=-A, b=b, c=-2.0))
            self.manifolds.append(PointManifold(goal=self.goal))

            self.obstacles = [[2, .5, 3., 1.5], [2, 3, .5, 1.5], [-2, .5, 3., 1.5], [-2, 3, .5, 1.5]]

    def get_joint_space_volume(self) -> float:
        """Returns the volume of the joint space defined by its lower and upper limits."""
        vol = 1.0
        for i in range(self.d):
            vol = vol * (self.lim_up[i] - self.lim_lo[i])
        return vol

    def is_collision_conf(self, q: np.ndarray) -> bool:
        """Checks if the configuration q is in collision."""
        for obs in self.obstacles:
            if np.fabs(q[2]-obs[0]) <= obs[3] and np.fabs(q[0]) <= obs[1] and np.fabs(q[1]) <= obs[2]:
                return True
        return False

    def is_valid_conf(self, q: np.ndarray) -> bool:
        """Checks if the configuration q fulfills the lower and upper configuration limits."""
        return check_limits(q, self.lim_lo, self.lim_up)

    def sample(self) -> np.ndarray:
        """Uniformly samples a point in the configuration limits."""
        q_target = np.empty(self.d)
        for i in range(self.d):
            q_target[i] = self.lim_lo[i] + np.random.rand() * (self.lim_up[i] - self.lim_lo[i])
        return q_target

    def plot(self,
             name: str,
             G_list: list = None,
             V_goal_list: list = None,
             opt_path: list = None):
        """Visualizes the tree, manifolds, paths with plotly in 3D."""
        colorscales = ['Reds', 'Greens', 'Blues', 'Magentas']
        color = ['red', 'green', 'blue', 'magenta']
        pd = []

        if self.d == 3:
            X = []
            Y = []
            Z = []
            if opt_path:
                for i, path in enumerate(opt_path):
                    X.clear(), Y.clear(), Z.clear()
                    for state in path:
                        X += [state[0]]
                        Y += [state[1]]
                        Z += [state[2]]
                    pd.append(go.Scatter3d(x=X, y=Y, z=Z, marker=dict(color=color[i], size=5), name='Path_M' + str(i)))

            if G_list:
                X.clear(), Y.clear(), Z.clear()
                for G in G_list:
                    for e in G.E.values():
                        X += [G.V[e.node_a].value[0], G.V[e.node_b].value[0], None]
                        Y += [G.V[e.node_a].value[1], G.V[e.node_b].value[1], None]
                        Z += [G.V[e.node_a].value[2], G.V[e.node_b].value[2], None]
                pd.append(go.Scatter3d(x=X, y=Y, z=Z, mode='lines', showlegend=True,
                                       line=dict(color='rgb(125,125,125)', width=0.5),
                                       hoverinfo='none', name='Tree'))
                pd.append(go.Scatter3d(x=[self.start[0]], y=[self.start[1]], z=[self.start[2]],
                                       mode='markers', marker=dict(color='red', size=5), name='Start'))

            if V_goal_list:
                X.clear(), Y.clear(), Z.clear()
                for i, V in enumerate(V_goal_list):
                    for j in V:
                        X += [G_list[i].V[j].value[0]]
                        Y += [G_list[i].V[j].value[1]]
                        Z += [G_list[i].V[j].value[2]]
                pd.append(go.Scatter3d(x=X, y=Y, z=Z, mode='markers',
                                       marker=dict(color='magenta', size=5),
                                       name='Intersection nodes'))

        if self.name in ['3d_point_wo_obstacles', '3d_point_w_obstacles']:
            for i, m in enumerate(self.manifolds):
                limits = [self.lim_lo[0], self.lim_up[0], self.lim_lo[1], self.lim_up[1]]
                X_m, Y_m, Z_m = m.draw(limits=limits)

                if m.draw_type == "Scatter":
                    pd.append(go.Scatter3d(x=X_m, y=Y_m, z=Z_m, showlegend=False, mode='markers',
                                           marker=dict(color=color[i], size=5)))
                elif m.draw_type == "Surface":
                    pd.append(go.Surface(x=X_m, y=Y_m, z=Z_m, opacity=0.8, showscale=False,
                                         colorscale=colorscales[i]))

        for obs in self.obstacles:
            plot_box(pd=pd, pos=np.array([0., 0., obs[0]]), quat=np.array([0., 0., 0., 1.]), size=np.array(obs[1:]))

        fig = go.Figure(data=pd, layout=go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1)))
        plot(fig, filename='plots/task_' + self.name + '_' + name + '.html', auto_open=True)
