import numpy as np
import configparser
from typing import Callable
from math import sqrt, pi, gamma
import os
import numbers
from scipy.spatial.transform.rotation import Rotation
import plotly.graph_objs as go


def create_dir(dirname: str):
    """Creates a dictionary if it does not already exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def read_cfg(cfg_name: str) -> dict:
    """Reads a config from text format into a dictionary."""
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


def check_limits(value,
                 min_value,
                 max_value) -> bool:
    """Check if all values are between [min_value, max_value].
    Three input types are possible:
    - value, min_value, max_value are scalars
    - value is an array and min_value, max_value are scalars
    - value, min_value, max_value are arrays and an elementwise check is performed
    """
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


def unit_ball_measure(n: int) -> float:
    """Computes volume of unit ball of dimension n."""
    return (sqrt(pi) ** n) / gamma(float(n) / 2.0 + 1.0)


def path_cost(path: list) -> float:
    """Computes length of a path."""
    cost = 0.0
    if type(path[0]) is list:
        for path_i in path:
            for i in range(1, len(path_i)):
                cost += np.linalg.norm(path_i[i-1] - path_i[i])
    else:
        for i in range(1, len(path)):
            cost += np.linalg.norm(path[i-1] - path[i])
    return cost


def is_on_manifold(m: Callable[[np.ndarray], np.ndarray],
                   q: np.ndarray,
                   eps=1e-4) -> bool:
    """Checks if a configuration q is on a manifold defined by a level set m(q)."""
    return np.linalg.norm(m.y(q)) < eps


def get_volume(low: list,
               up: list) -> float:
    """Returns the volume defined of a box defined by lower and upper limits."""
    vol = 1.0
    for i in range(len(low)):
        vol = vol * (up[i] - low[i])
    return vol


def plot_box(pd: list,
             pos: np.ndarray,
             quat: np.ndarray,
             size: np.ndarray):
    """Plots a box in plotly at location pos with rotation quat and dimensions size."""
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
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        name='y',
        showscale=False
        )
    )


class Projection:
    """This class implements an iterative optimization routine that projects configurations onto a constraint."""
    def __init__(self,
                 f: Callable[[np.ndarray], np.ndarray],
                 J: Callable[[np.ndarray], np.ndarray],
                 tol: float = 1e-5,
                 max_iter: int = 200,
                 step_size: float = 1.0):
        self.f = f
        self.J = J
        self.tol = tol
        self.max_iter = max_iter
        self.step_size = step_size

    def project(self, q: np.ndarray) -> (bool, np.ndarray):
        """Projects a point onto the constraint and return a boolean indicating success and the projected point."""
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
