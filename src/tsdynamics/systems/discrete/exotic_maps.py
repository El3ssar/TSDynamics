from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Bogdanov(DynMap):
    params = {
            "eps": 0.0,
            "k": 1.2,
            "mu": 0.0
        }
    n_dim = 2
    @staticjit
    def _rhs(X, eps, k, mu):
        x, y = X
        yp = (1 + eps) * y + k * x * (x - 1) + mu * x * y
        xp = x + yp
        return xp, yp
    
    @staticjit
    def _jac(X, eps, k, mu):
        x, y = X
        row1 = [1, 1 + eps + mu * x]
        row2 = [2 * k * x - k * x + mu * y, 1 + eps + mu * x]
        return row1, row2


class Svensson(DynMap):
    params = {
            "a": 1.5, 
            "b": -1.8, 
            "c": 1.6, 
            "d": 0.9
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b, c, d):
        x, y = X
        xp = d * np.sin(a * x) - np.sin(b * y)
        yp = c * np.cos(a * x) + np.cos(b * y)
        return xp, yp

    @staticjit
    def _jac(X, a, b, c, d):
        x, y = X
        row1 = [a * d * np.cos(a * x), -b * np.cos(b * y)]
        row2 = [-a * c * np.sin(a * x), -b * np.sin(b * y)]
        return row1, row2

class Bedhead(DynMap):
    params = {
            "a": -0.67,
            "b":  0.83
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b):
        x, y = X
        xp = np.sin(x * y / b) * y + np.cos(a * x - y)
        yp = x + np.sin(y) / b
        return xp, yp

    @staticjit
    def _jac(X, a, b):
        x, y = X
        row1 = [y * np.cos(x * y / b) / b - a * np.sin(a * x - y), np.sin(x * y / b) + y * np.cos(a * x - y) + np.sin(a * x - y)]
        row2 = [1, np.cos(y) / b]
        return row1, row2

class ZeraouliaSprott(DynMap):
    params = {
            "a": 1.641,
            "b": 1.902
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b):
        x, y = X
        xp = - a * x / (1 + y**2)
        yp = x + b * y
        return xp, yp
    
    @staticjit
    def _jac(X, a, b):
        x, y = X
        row1 = [-a / (1 + y**2), 2 * a * x * y / (1 + y**2)**2]
        row2 = [1, b]
        return row1, row2


class GumowskiMira(DynMap):
    params = {
            "a": -1.1,
            "b": -0.2
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b):
        x, y = X
        fx = a * x + 2 * (1 - a) * x**2 / (1 + x**2)
        xp = b * y + fx
        fx1 = a * xp + 2 * (1 - a) * xp**2 / (1 + xp**2)
        yp = fx1 - x
        return xp, yp

    @staticjit
    def _jac(X, a, b):
        pass

class Hopalong(DynMap):
    params = {
            "a": 3.1,
            "b": 2.5,
            "c": 4.2
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b, c):
        x, y = X
        xp = y - 1 - np.sqrt(np.abs(b * x - 1 - c)) * np.sign(x - 1)
        yp = a - x - 1
        return xp, yp
    
    @staticjit
    def _jac(X, a, b, c):
        pass

class Pickover(DynMap):
    params = {
            "a": -1.4,
            "b": 1.6,
            "c": 1.0,
            "d": 0.7
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) + c * np.cos(a * x)
        yp = np.sin(b * x) + d * np.cos(b * y)
        return xp, yp
    
    @staticjit
    def _jac(X, a, b, c, d):
        x, y = X
        row1 = [-a * c * np.sin(a * x), a * np.cos(a * y)]
        row2 = [b * np.cos(b * x), -b * d * np.sin(b * y)]
        return row1, row2

