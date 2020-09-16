import numpy as np
import configparser
from math import sqrt, pi, gamma
import os
import numbers
from scipy.spatial.transform.rotation import Rotation
import plotly.graph_objs as go
from plotly.offline import plot


def read_cfg(cfg_name):
    cfg = configparser.ConfigParser()
    if cfg_name[-3:] == 'cfg':
        cfg.read(cfg_name)
    else:
        cfg.read_string(cfg_name)

    c = dict()
    c['SEED'] = cfg.getint('general', 'SEED')
    c['CONV_TOL'] = cfg.getfloat('general', 'CONV_TOL')
    c['N'] = cfg.getint('general', 'N')
    c['ALPHA'] = cfg.getfloat('general', 'ALPHA', fallback=1.0)
    c['BETA'] = cfg.getfloat('general', 'BETA', fallback=0.0)
    c['COLLISION_RES'] = cfg.getfloat('general', 'COLLISION_RES', fallback=1.0)
    c['EPS'] = cfg.getfloat('general', 'EPS', fallback=1e-2)
    c['RHO'] = cfg.getfloat('general', 'RHO', fallback=5e-1)
    c['R_MAX'] = cfg.getfloat('general', 'R_MAX', fallback=1e-1)
    c['GREEDY'] = cfg.getboolean('general', 'GREEDY', fallback=False)
    c['PROJ_STEP_SIZE'] = cfg.getfloat('general', 'PROJ_STEP_SIZE', fallback=1.0)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    print(c)
    return c


def jac_check(x, f, dfdx, eps=1e-7, tol=1e-4):
    y = f(x)
    J = dfdx(x)

    assert len(J.shape) == 2

    [n, m] = J.shape
    assert m == x.shape[0]
    if not np.isscalar(y):
        assert n == y.shape[0]

    J_eps = np.empty((n, m))

    for i in range(m):
        x_i = np.copy(x)
        x_i[i] += eps
        y_i = f(x_i)
        df = (y_i - y) / eps

        if np.isscalar(y):
            J_eps[0, i] = df
        else:
            for j in range(n):
                J_eps[j, i] = df[j]

    jac_diff = np.abs(J_eps - J)
    if np.max(jac_diff) > tol:
        print("analytic and numeric Jacobian do not match")
        print("analytic Jacobian ", J)
        print("numeric Jacobian  ", J_eps)
        print("max abs error ", np.max(jac_diff))
        return False

    return True


def check_limits(value, min_value, max_value):
    # value, max and min are arrays
    if (type(value) is np.ndarray) and type(max_value) is np.ndarray and type(min_value) is np.ndarray:
        for i in range(value.size):
            if value.item(i) > max_value.item(i) or value.item(i) < min_value.item(i):
                return False
        return True

    # value is array, max and min are floats
    if (type(value) is np.ndarray) and type(max_value) is float and type(min_value) is float:
        for i in range(value.size):
            if value.item(i) > max_value or value.item(i) < min_value:
                return False
        return True

    # value, max, min are floats
    if isinstance(value, numbers.Number):
        return value <= max_value and value >= min_value

    raise TypeError("inputs to limit_value have to be arrays or floats, but is", type(value))


def unit_ball_measure(n):
    return (sqrt(pi) ** n) / gamma(float(n) / 2.0 + 1.0)


def path_cost(path):
    cost = 0.0
    for i in range(1, len(path)):
        cost += np.linalg.norm(path[i-1] - path[i])
    return cost


def is_on_manifold(m, q, eps=1e-4):
    return np.linalg.norm(m.y(q)) < eps


def plot_box(pd, pos, quat, size):
    d = -size
    p = size
    X = np.array([[d[0], d[0], p[0], p[0], d[0], d[0], p[0], p[0]],
                  [d[1], p[1], p[1], d[1], d[1], p[1], p[1], d[1]],
                  [d[2], d[2], d[2], d[2], p[2], p[2], p[2], p[2]]])

    R = Rotation.from_quat(quat)
    X = R.apply(X.T) + pos

    pd.append(go.Mesh3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        flatshading=True,
        lighting=dict(facenormalsepsilon=0),
        lightposition=dict(x=2000, y=1000),
        color='black',
        # i, j and k give the vertices of triangles
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        name='y',
        showscale=False
        )
    )


class Projection:
    def __init__(self, f_, J_, tol_=1e-5, max_iter_=200, step_size_=1.0):
        self.f = f_
        self.J = J_
        self.tol = tol_
        self.max_iter = max_iter_
        self.step_size = step_size_

    def project(self, q):
        y = self.f(q)
        y0 = 2.0 * np.linalg.norm(y)
        iter = 0
        while np.linalg.norm(y) > self.tol and iter < self.max_iter and np.linalg.norm(y) < y0:
            J = self.J(q)
            q = q - self.step_size * np.linalg.lstsq(J, y, rcond=-1)[0]
            y = self.f(q)

            iter += 1

        result = np.linalg.norm(y) <= self.tol
        return result, np.array(q)
