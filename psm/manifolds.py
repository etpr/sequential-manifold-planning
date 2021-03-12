import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    """Abstract class to represent a manifold."""
    @abstractmethod
    def __init__(self,
                 name: str,
                 dim_ambient: int,
                 dim_manifold: int):
        self.name = name
        self.dim_ambient = dim_ambient
        self.dim_manifold = dim_manifold

    """This function describes the level set of the manifold y(x)=0."""
    @abstractmethod
    def y(self, x: np.ndarray) -> np.ndarray:
        pass

    """The Jacobian of the level set function."""
    @abstractmethod
    def J(self, x: np.ndarray) -> np.ndarray:
        pass

    """Draw routines to visualize manifolds in 2D/3D."""
    @abstractmethod
    def draw(self, limits):
        pass

    @property
    @abstractmethod
    def draw_type(self):
        pass


class PointManifold(Manifold):
    def __init__(self, goal: np.ndarray):
        Manifold.__init__(self, name="Point", dim_ambient=goal.shape[0], dim_manifold=goal.shape[0])
        self.goal = goal

    def y(self, x: np.ndarray) -> np.ndarray:
        return self.goal - x

    def J(self, x: np.ndarray) -> np.ndarray:
        return -np.eye(self.goal.shape[0])

    def draw(self, limits):
        return [self.goal[0]], [self.goal[1]], [self.goal[2]]

    @property
    def draw_type(self):
        return "Scatter"


class PlaneManifold(Manifold):
    def __init__(self,
                 a: float,
                 b: float,
                 c: float,
                 d: float):
        Manifold.__init__(self, name="Plane", dim_ambient=3, dim_manifold=1)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[self.a, self.b, self.c]])

    def draw(self, limits):
        x_min, x_max, y_min, y_max = limits
        if np.abs(self.c) > 0:
            n = 100
            X = np.linspace(x_min, x_max, n)
            Y = np.linspace(y_min, y_max, n)
            X, Y = np.meshgrid(X, Y)
            f = lambda x: -(self.a * x[0] + self.b * x[1] + self.d) / self.c
            Z = [f(np.array([x, y])) for (x, y) in np.nditer([X, Y])]
            Z = np.asarray(Z).reshape([n, n])
            return X, Y, Z
        else:
            pass

    @property
    def draw_type(self):
        return "Surface"


class EllipsoidManifold(Manifold):
    def __init__(self,
                 a: float,
                 b: float,
                 c: float):
        Manifold.__init__(self, name="Ellipsoid", dim_ambient=3, dim_manifold=1)
        self.a = a
        self.b = b
        self.c = c

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 / self.a ** 2 + x[1] ** 2 / self.b ** 2 + x[2] ** 2 / self.c ** 2 - 1.0])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0] / self.a ** 2,
                          2 * x[1] / self.b ** 2,
                          2 * x[2] / self.c ** 2]])

    def param_to_xyz(self, param: list):
        theta = param[0]
        phi = param[1]
        if np.isscalar(theta):
            x = self.a * np.cos(theta) * np.sin(phi)
            y = self.b * np.sin(theta) * np.sin(phi)
            z = self.c * np.cos(phi)
        else:
            x = self.a * np.outer(np.cos(theta), np.sin(phi))
            y = self.b * np.outer(np.sin(theta), np.sin(phi))
            z = self.c * np.outer(np.ones_like(theta), np.cos(phi))

        return x, y, z

    def draw(self, limits):
        n = 100
        theta = np.linspace(0, 2 * np.pi, n)
        phi = np.linspace(0, np.pi, n)

        X, Y, Z = self.param_to_xyz([theta, phi])
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class SphereManifold(Manifold):
    def __init__(self, r: float):
        Manifold.__init__(self, name="Sphere", dim_ambient=3, dim_manifold=1)
        self.r = r

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.dot(x, x) - self.r ** 2])

    def J(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * np.array([[x[0], x[1], x[2]]])

    def param_to_xyz(self, param: list):
        theta = param[0]
        phi = param[1]
        if np.isscalar(theta):
            x = self.r * np.cos(theta) * np.sin(phi)
            y = self.r * np.sin(theta) * np.sin(phi)
            z = self.r * np.cos(phi)
        else:
            x = self.r * np.outer(np.cos(theta), np.sin(phi))
            y = self.r * np.outer(np.sin(theta), np.sin(phi))
            z = self.r * np.outer(np.ones_like(theta), np.cos(phi))

        return x, y, z

    def draw(self, limits):
        n = 100
        theta = np.linspace(0, 2 * np.pi, n)
        phi = np.linspace(0, np.pi, n)

        X, Y, Z = self.param_to_xyz([theta, phi])
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class ParaboloidManifold(Manifold):
    def __init__(self,
                 A: np.ndarray,
                 b: np.ndarray,
                 c: float):
        Manifold.__init__(self, name="Paraboloid", dim_ambient=3, dim_manifold=1)
        self.A = A
        self.b = b
        self.c = c

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.transpose(x[:2]) @ self.A @ x[:2] + np.dot(self.b, x[:2]) + self.c - x[2]])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2.0 * self.A[0, 0] * x[0] + self.A[0, 1] * x[1] + self.A[1, 0] * x[1] + self.b[0],
                          2.0 * self.A[1, 1] * x[1] + self.A[0, 1] * x[0] + self.A[1, 0] * x[0] + self.b[1],
                          -1]])

    def draw(self, limits):
        x_min, x_max, y_min, y_max = limits
        n = 100
        X = np.linspace(x_min, x_max, n)
        Y = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(X, Y)
        f = lambda x: np.matmul(x.transpose(), np.matmul(self.A, x)) + np.dot(self.b, x) + self.c
        Z = [f(np.array([x, y])) for (x, y) in np.nditer([X, Y])]
        Z = np.asarray(Z).reshape([n, n])
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class CylinderManifold(Manifold):
    def __init__(self,
                 a: float,
                 b: float):
        Manifold.__init__(self, name="Cylinder", dim_ambient=3, dim_manifold=1)
        self.a = a
        self.b = b

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 / (self.a ** 2) + x[1] ** 2 / (self.b ** 2) - 1.0])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2.0 * x[0] / (self.a ** 2),
                          2.0 * x[1] / (self.b ** 2),
                          0.0]])

    def draw(self, limits):
        n = 50
        U = np.linspace(0, 2 * np.pi, n)
        Z = np.linspace(-5, 5, n)
        U, Z = np.meshgrid(U, Z)
        X = self.a * np.cos(U)
        Y = self.b * np.sin(U)
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class LineManifold(Manifold):
    def __init__(self,
                 p: np.ndarray,
                 v: np.ndarray):
        Manifold.__init__(self, name="Line", dim_ambient=3, dim_manifold=2)
        self.p = p
        self.v = v

        # find two vectors (c1, c2) orthogonal to v
        if self.v[0] == 0:
            self.c1 = np.array([0, self.v[2], -self.v[1]])
        elif self.v[1] == 0:
            self.c1 = np.array([-self.v[2], 0, self.v[0]])
        elif self.v[2] == 0:
            self.c1 = np.array([-self.v[1], self.v[0], 0])
        else:
            self.c1 = np.array([0, self.v[2], -self.v[1]])

        self.c2 = np.cross(self.v, self.c1)

    def y(self, x: np.ndarray) -> np.ndarray:
        d = x - self.p
        return np.array([np.dot(self.c1, d), np.dot(self.c2, d)])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.array([self.c1[0], self.c1[1], self.c1[2]]),
                         np.array([self.c2[0], self.c2[1], self.c2[2]])])

    def draw(self, limits):
        n = 100
        T = np.linspace(0, 10, n)
        f = lambda t: self.p + t * self.v
        M = [f(t) for t in T]

        X = [x[0] for x in M]
        Y = [x[1] for x in M]
        Z = [x[2] for x in M]

        return X, Y, Z

    @property
    def draw_type(self):
        return "Scatter"


class PeriodicManifold(Manifold):
    def __init__(self,
                 a: float,
                 b: float,
                 c: float,
                 d: float):
        Manifold.__init__(self, name="Periodic", dim_ambient=3, dim_manifold=1)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([(np.sin(x[0] * np.pi * self.a) + x[0] * self.b + x[1] * self.c) / self.d - x[2]])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[(np.pi * self.a * np.cos(x[0] * np.pi * self.a) + self.b) / self.d,
                          self.c / self.d,
                          -1.0]])

    def draw(self, limits):
        x_min, x_max, y_min, y_max = limits
        n = 100
        X = np.linspace(x_min, x_max, n)
        Y = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(X, Y)
        f = lambda x: (np.sin(x[0] * np.pi * self.a) + x[0] * self.b + x[1] * self.c) / self.d
        Z = [f(np.array([x, y])) for (x, y) in np.nditer([X, Y])]
        Z = np.asarray(Z).reshape([n, n])
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class LoopManifold(Manifold):
    def __init__(self, r: float):
        Manifold.__init__(self, name="Loop", dim_ambient=3, dim_manifold=2)
        self.r = r

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1] ** 2 - self.r ** 2,
                         x[2]])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0], 2 * x[1], 0.0],
                         [0.0, 0.0, 1.0]])

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Scatter"


class FreeManifold(Manifold):
    def __init__(self, d: int):
        Manifold.__init__(self, name="Free", dim_ambient=d, dim_manifold=0)
        self.d = d

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(1)

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((1, self.d))

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Scatter"


class CircleManifold(Manifold):
    def __init__(self, r: float):
        Manifold.__init__(self, name="Circle", dim_ambient=2, dim_manifold=1)
        self.r = r

    def y(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1] ** 2 - self.r ** 2])

    def J(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0], 2 * x[1]]])

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Scatter"


class ManifoldStack(Manifold):
    def __init__(self, manifolds):
        self.manifolds = manifolds
        dim_manifold = 0
        for f in self.manifolds:
            dim_manifold += f.dim_manifold
        Manifold.__init__(self, name="ManifoldStack", dim_ambient=3, dim_manifold=dim_manifold)

    def y(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(0)
        for f in self.manifolds:
            out = np.append(out, f.y(x))
        return out

    def J(self, x: np.ndarray) -> np.ndarray:
        out = np.empty((0, len(x)))
        for f in self.manifolds:
            out = np.append(out, f.J(x), axis=0)
        return out

    def draw(self, limits):
        X = []
        Y = []
        Z = []
        for f in self.manifolds:
            draw_op = getattr(f, "draw", None)
            if callable(draw_op):
                Xf, Yf, Zf = f.draw(limits)

                X = Xf
                Y = Yf
                Z = Zf
        return X, Y, Z

    @property
    def draw_type(self):
        return "Stacked"
