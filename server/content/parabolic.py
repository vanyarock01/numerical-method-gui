#!/usr/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
from py_expression_eval import Parser
import pprint

class Solver(object):

    def __init__(self, args):
        for arg_name, arg_value in args.items():
            setattr(self, arg_name, arg_value)

        self.math_parser = Parser()
        
        # set grid
        self.U = np.ndarray( (self.K + 1, self.N + 1) )
        self.U.fill(0)
        
        # create vectors
        self.vecX = np.linspace(0, self.l, self.N + 1)
        self.vecT = np.linspace(0, self.T, self.K + 1)

        # set grid steps
        self.h = self.l / self.N
        self.tau = self.T / self.K

        if not hasattr(self, 'exact_solution'):
            self.exact_solution = ''
        else:
            self.reference_grid = np.ndarray((self.K + 1, self.N + 1))
            self.reference_grid.fill(0)
        
    def getResultsDict(self):
        self.solve()
        res = {
            'x': self.vecX,
            't': self.vecT,
            'U': self.U
        }
        if self.exact_solution != '':
            self.calc_reference_grid()
            res['exact_solution'] = self.reference_grid
        return res

    def solve(self):
        self.presolveSetup()
        for i in range(self.K):
            self.resetTmp()
            tNext = self.vecT[i + 1]
            self._phi0 = self.execLeftBound(tNext)
            self._phiN = self.execRightBound(tNext)

            for j in range(1, self.N):
                self.res = 0.0

                self.res += (self.a / (self.h ** 2)) * \
                    (self.U[i, j + 1] - 2.0 * self.U[i, j] + self.U[i, j - 1])
                self.res += self.b / (2.0 * self.h) * \
                    (self.U[i, j + 1] - self.U[i, j - 1])
                self.res += self.c * self.U[i, j]
                self.res = -(1.0 - self.theta) * self.res
                self.res -= self.U[i, j] / self.tau
                self.res -= self.execEquationFunction(self.vecX[j], tNext)

                self.mat[j, j - 1] = self.A
                self.mat[j, j] = self.B
                self.mat[j, j + 1] = self.C
                self.vec[j] = self.res

            if self.bound_cond_type == "2-1":
                self.bound_cond_2_1()
            elif self.bound_cond_type == "2-2":
                self.bound_cond_2_2(i)
            elif self.bound_cond_type == "3-2":
                self.bound_cond_3_2()

            self.vecRes = np.linalg.solve(self.mat, self.vec)

            for j in range(self.N + 1):
                self.U[i + 1, j] = self.vecRes[j]

    def bound_cond_2_1(self):
        self.mat[0, 0] = self.beta - self.alpha / self.h
        self.mat[0, 1] = self.alpha / self.h
        self.mat[self.N, self.N - 1] = -self.gamma / self.h
        self.mat[self.N, self.N] = self.delta + self.gamma / self.h
        self.vec[0] = self._phi0
        self.vec[self.N] = self._phiN

    def bound_cond_2_2(self, i):
        if (self.alpha == 0):
            self.mat[0, 0] = self.beta
            self.vec[0] = self._phi0
        else:
            b0 = 2.0 * self.a / self.h + self.h / self.tau - self.c * self.h - \
                (self.beta / self.alpha) * (2 * self.a - self.b * self.h)
            c0 = -2.0 * self.a / self.h
            d0 = (self.h / self.tau) * \
                self.U[i, 0] - self._phi0 * \
                (2 * self.a - self.b * self.h) / self.alpha

            self.mat[0, 0] = b0
            self.mat[0, 1] = c0
            self.vec[0] = d0

        if (self.gamma == 0):
            self.mat[self.N, self.N] = self.delta
            self.vec[self.N] = self._phiN
        else:
            aN = -2.0 * self.a / self.h
            bN = 2.0 * self.a / self.h + self.h / self.tau - self.c * self.h + \
                (self.delta / self.gamma) * (2 * self.a + self.b * self.h)
            dN = (self.h / self.tau) * self.U[i, self.N] + \
                self._phiN * (2 * self.a + self.b * self.h) / self.gamma

            self.mat[self.N, self.N - 1] = aN
            self.mat[self.N, self.N] = bN
            self.vec[self.N] = dN

    def bound_cond_3_2(self):
        h2 = 2 * self.h
        self.mat[0, 0] = self.beta - 3.0 * self.alpha / h2
        self.mat[0, 1] = 4.0 * self.alpha / h2
        self.mat[0, 2] = -(self.alpha / h2)
        self.mat[self.N, self.N - 2] = self.gamma / h2
        self.mat[self.N, self.N - 1] = -4.0 * self.gamma / h2
        self.mat[self.N, self.N] = self.delta + 3.0 * self.gamma / h2
        self.vec[0] = self._phi0
        self.vec[self.N] = self._phiN

    def presolveSetup(self):
        self.setInitialVectorX()
        self.setTheta()
        self.setSigmas()
        self.setCoefsSLE()
        self.createTmp()

    def createTmp(self):
        self.mat = np.ndarray((self.N + 1, self.N + 1))
        self.vec = np.ndarray((self.N + 1))
        self.vecRes = np.ndarray((self.N + 1))

    def resetTmp(self):
        self.mat.fill(0)
        self.vec.fill(0)
        self.vecRes.fill(0)

    def setInitialVectorX(self):
        for j in range(self.N + 1):
            self.U[0, j] = self.execInitialFunction(self.vecX[j])

    def execInitialFunction(self, x):
        return self.math_parser.parse(self.psi).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'x': x
        })

    def setTheta(self):
        if not self.checkSigma() and self.scheme_type == "EXPLICIT":
            print("SIGMA ERROR")
            return self.U
        if self.scheme_type == "EXPLICIT":
            self.theta = 0.0
        elif self.scheme_type == "IMPLICIT":
            self.theta = 1.0
        elif self.scheme_type == "CRANK_NICOLSON":
            self.theta = 0.5

    def checkSigma(self):
        return self.a * self.tau / (self.h ** 2) < 0.5

    def setSigmas(self):
        self.sigma1 = self.theta * self.a / (self.h ** 2)
        self.sigma2 = self.theta * self.c
        self.sigma3 = self.theta * self.b / (2 * self.h)

    def setCoefsSLE(self):
        self.A = self.sigma1 - self.sigma2
        self.B = self.sigma2 - 2.0 * self.sigma1 - 1.0 / self.tau
        self.C = self.sigma1 + self.sigma3

    def execLeftBound(self, t):
        return self.math_parser.parse(self.phi_0).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            't': t,
        })

    def execRightBound(self, t):
        return self.math_parser.parse(self.phi_n).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            't': t,
        })

    def execEquationFunction(self, x, t):
        return self.math_parser.parse(self.f).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            't': t,
            'x': x
        })

    def calc_reference_grid(self):
        for i in range(self.K + 1):
            for j in range(self.N + 1):
                self.reference_grid[i, j] = self.exec_reference(
                    self.vecX[j], self.vecT[i])

    def exec_reference(self, x, t):
        return self.math_parser.parse(self.exact_solution).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'x': x,
            't': t,
        })


if __name__ == "__main__":
    args = {
        'a': 0.001,
        'b': 0,
        'c': 0,
        'l': 1.0,
        'alpha': 0,
        'beta': 1,
        'gamma': 0,
        'delta': 1,
        'T': 50,
        'K': 100,
        'N': 10,
        'f': '0',
        'bound_cond_type': 1,
        'scheme_type': 'EXPLICIT',
        'psi': 'sin(2 * PI * x)',
        'phi_0': '0',
        'phi_n': '0',
        'exact_solution': 'exp(-4 * pow(PI, 2) * a * t) * sin(2 * PI * x)'
    }
    
    solve = Solver(args)

    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(exact_solution)
    fig.add_subplot(1, 2, 2)
    plt.imshow(solve.getResultsDict()['U'])
    plt.show()
