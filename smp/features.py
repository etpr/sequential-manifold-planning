import numpy as np
from abc import ABC, abstractmethod


class Feature(ABC):
    @abstractmethod
    def __init__(self, name, dim_ambient, dim_feature):
        self.name = name
        self.dim_ambient = dim_ambient
        self.dim_feature = dim_feature
        self.is_inequality = False

    @abstractmethod
    def y(self, x):
        pass

    @abstractmethod
    def J(self, x):
        pass

    @abstractmethod
    def draw(self, limits):
        pass

    @property
    @abstractmethod
    def draw_type(self):
        pass


class PointFeature(Feature):
    def __init__(self, goal):
        Feature.__init__(self, "Point", dim_ambient=goal.shape[0], dim_feature=goal.shape[0])
        self.goal = goal

    def y(self, x):
        return self.goal - x

    def J(self, x):
        return -np.eye(self.goal.shape[0])

    def draw(self, limits):
        return [self.goal[0]], [self.goal[1]], [self.goal[2]]

    @property
    def draw_type(self):
        return "Scatter"


class PlaneFeature(Feature):
    def __init__(self, a, b, c, d):
        Feature.__init__(self, "Plane", dim_ambient=3, dim_feature=1)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def y(self, x):
        return np.array([self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d])

    def J(self, x):
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


class EllipsoidFeature(Feature):
    def __init__(self, a, b, c):
        Feature.__init__(self, "Ellipsoid", dim_ambient=3, dim_feature=1)
        self.a = a
        self.b = b
        self.c = c

    def y(self, x):
        return np.array([x[0] ** 2 / self.a ** 2 + x[1] ** 2 / self.b ** 2 + x[2] ** 2 / self.c ** 2 - 1.0])

    def J(self, x):
        return np.array([[2 * x[0] / self.a ** 2,
                          2 * x[1] / self.b ** 2,
                          2 * x[2] / self.c ** 2]])

    def param_to_xyz(self, param):
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


class SphereFeature(Feature):
    def __init__(self, r):
        Feature.__init__(self, "Sphere", dim_ambient=3, dim_feature=1)
        self.r = r

    def y(self, x):
        return np.array([np.dot(x, x) - self.r ** 2])

    def J(self, x):
        return 2.0 * np.array([[x[0], x[1], x[2]]])

    def param_to_xyz(self, param):
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


class ParaboloidFeature(Feature):
    def __init__(self, A, b, c):
        Feature.__init__(self, "Paraboloid", dim_ambient=3, dim_feature=1)
        self.A = A
        self.b = b
        self.c = c

    def y(self, x):
        return np.array([np.transpose(x[:2]) @ self.A @ x[:2] + np.dot(self.b, x[:2]) + self.c - x[2]])

    def J(self, x):
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


class CylinderFeature(Feature):
    def __init__(self, a, b):
        Feature.__init__(self, "Cylinder", dim_ambient=3, dim_feature=1)
        self.a = a
        self.b = b

    def y(self, x):
        return np.array([x[0] ** 2 / (self.a ** 2) + x[1] ** 2 / (self.b ** 2) - 1.0])

    def J(self, x):
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


class LineFeature(Feature):
    def __init__(self, p, v):
        Feature.__init__(self, "Line", dim_ambient=3, dim_feature=2)
        self.p = p
        self.v = v

        # searching for two vectors orthogonal to v
        if self.v[0] is 0:
            self.c1 = np.array([0, self.v[2], -self.v[1]])
        elif self.v[1] is 0:
            self.c1 = np.array([-self.v[2], 0, self.v[0]])
        elif self.v[2] == 0:
            self.c1 = np.array([-self.v[1], self.v[0], 0])
        else:
            self.c1 = np.array([0, self.v[2], -self.v[1]])

        self.c2 = np.cross(self.v, self.c1)

    def y(self, x):
        d = x - self.p
        return np.array([np.dot(self.c1, d), np.dot(self.c2, d)])

    def J(self, x):
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


class PeriodicFeature(Feature):
    def __init__(self, a, b, c, d):
        Feature.__init__(self, "Periodic", dim_ambient=3, dim_feature=1)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def y(self, x):
        return np.array([(np.sin(x[0] * np.pi * self.a) + x[0] * self.b + x[1] * self.c) / self.d - x[2]])

    def J(self, x):
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


class LoopFeature(Feature):
    def __init__(self, r):
        Feature.__init__(self, "Loop", dim_ambient=3, dim_feature=2)
        self.r = r

    def y(self, x):
        return np.array([x[0]**2 + x[1]**2 - self.r ** 2,
                         x[2]])

    def J(self, x):
        return np.array([[2*x[0], 2*x[1], 0.0],
                         [0.0, 0.0, 1.0]])

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Scatter"


class FeatureStack(Feature):
    def __init__(self, features_):
        self.features = features_
        dim_feature = 0
        for f in self.features:
            dim_feature += f.dim_feature
        Feature.__init__(self, "FeatureStack", dim_ambient=3, dim_feature=dim_feature)

    def y(self, q):
        out = np.empty(0)
        for f in self.features:
            out = np.append(out, f.y(q))
        return out

    def J(self, q):
        out = np.empty((0, len(q)))
        for f in self.features:
            out = np.append(out, f.J(q), axis=0)
        return out

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Stacked"
