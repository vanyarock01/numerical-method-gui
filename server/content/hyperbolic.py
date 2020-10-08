import math
import numpy as np
import matplotlib.pyplot as plt
from py_expression_eval import Parser
from pprint import pprint

class Solver(object):

    def __init__(self, args):
        pprint(args)
        for arg_name, arg_value in args.items():
            setattr(self, arg_name, arg_value)

        self.mathParser = Parser()
        self.parseArgs(args)
        self.createGrid()
        self.createVectors()
        self.setGridSteps()
        self.setReference(args)

    def parseArgs(self, args):
        self.parseEquation(args)
        self.parseBound(args)
        self.parseInitialCond(args)

    def parseEquation(self, args):
        self.parseEquationCoefs(args)
        self.parseEquationFunction(args)
        self.parseEquationParams(args)

    def parseEquationCoefs(self, args):
        self.a = args['a']
        self.b = args['b']
        self.c = args['c']
        self.e = args['e']

    def parseEquationFunction(self, args):
        self.f = args['f']

    def parseEquationParams(self, args):
        self.boundCondType = args['boundCondType']
        self.schemeType = args['schemeType']
        self.initCondType = args['initCondType']
        self.parseParamsT(args)
        self.parseParamsX(args)

    def parseParamsT(self, args):
        self.T = args['T']
        self.K = args['K']

    def parseParamsX(self, args):
        self.l = args['l']
        self.N = args['N']

    def parseBound(self, args):
        self.parseLeftBound(args)
        self.parseRightBound(args)

    def parseLeftBound(self, args):
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.phi0 = args['phi0']

    def parseRightBound(self, args):
        self.gamma = args['gamma']
        self.delta = args['delta']
        self.phiN = args['phiN']

    def parseInitialCond(self, args):
        self.psi1 = args['psi1']
        self.psi2 = args['psi2']
        self.psi1Derivative1 = args['psi1Derivative1']
        self.psi1Derivative2 = args['psi1Derivative2']

    def createGrid(self):
        self.U = np.ndarray((self.K + 1, self.N + 1))
        self.U.fill(0)

    def createVectors(self):
        self.vecX = np.linspace(0, self.l, self.N + 1)
        self.vecT = np.linspace(0, self.T, self.K + 1)

    def setGridSteps(self):
        self.h = self.l / self.N
        self.tau = self.T / self.K

    def setReference(self, args):
        if not 'reference' in args:
            self.reference = ''
            return
        self.reference = args['reference']
        self.referenceGrid = np.ndarray((self.K + 1, self.N + 1))
        self.referenceGrid.fill(0)

    def getResultsDict(self):
        self.solve()
        res =  {
            'x': self.vecX,
            't': self.vecT,
            'U': self.U
        }
        if self.reference != '':
            self.calcReferenceGrid()
            res['exact_solution'] = self.referenceGrid
        return res

    def solve(self):
        self.setInitialVecX()
        self.createTmp()
        self.setCoefsSLE()
        for i in range(1, self.K):
            self.tNext = self.vecT[i + 1]
            self._phi0 = self.execLeftBound(self.tNext)
            self._phi1 = self.execRightBound(self.tNext)

            if self.schemeType == 'EXPLICIT':
                self.explicit(i)

            elif self.schemeType == 'IMPLICIT':
                self.implicit(i)

    def explicit(self, i):
        for j in range(1, self.N):
            res = 0.0
            res += self.a ** 2 * (self.U[i, j + 1] - 2.0 * self.U[i, j] + self.U[i, j - 1]) / (self.h ** 2)
            res += self.b * (self.U[i, j + 1] - self.U[i, j - 1]) / (2.0 * self.h)
            res += self.c * self.U[i, j] + self.execEquationFunction(self.vecX[j], self.vecT[i])
            res *= 2.0 * (self.tau ** 2)
            res -= 2.0 * (self.U[i - 1, j] - 2.0 * self.U[i, j])
            res += self.e * self.tau * self.U[i - 1, j]
            res /= 2.0 + self.e * self.tau
            self.U[i + 1, j] = res

        bound1 = 0.0
        bound2 = 0.0

        if self.boundCondType == '2_1':
            bound1 = (self.h * self._phi0 - self.alpha * self.U[i + 1, 1]) / (self.h * self.beta - self.alpha)
            bound2 = (self.h * self._phi1 + self.gamma * self.U[i + 1, self.N - 1]) / (self.h * self.delta + self.gamma)

        elif self.boundCondType == '3_2':
            bound1 = 2.0 * self.h * self._phi0 - self.alpha * (4 * self.U[i + 1, 1] - self.U[i + 1, 2])
            bound1 /= 2.0 * self.h * self.beta - 3 * self.alpha

            bound2 = 2.0 * self.h * self._phi1 - self.gamma * (self.U[i + 1, self.N - 2] - 4 * self.U[i + 1, self.N - 1])
            bound2 /= 2.0 * self.h * self.delta + 3 * self.gamma

        elif self.boundCondType == '2_2':
            if (self.alpha == 0):
                bound1 = self._phi0 / self.beta
            else:
                b0 = 2.0  * self.a ** 2 / self.h + self.h / self.tau - self.c * self.h - (self.beta / self.alpha) * (2.0 * self.a ** 2 - self.b * self.h)
                c0 = -2.0 * self.a ** 2 / self.h
                d0 = (self.h / self.tau) * self.U[i, 0] - self._phi0 * (2.0 * self.a ** 2 - self.b * self.h) / self.alpha

                bound1 = (d0 - c0 * self.U[i + 1, 1]) / b0

            if self.gamma == 0:
                bound2 = self._phi1 / self.delta
            else:
                an = -2.0 * self.a ** 2 / self.h
                bn = 2.0 * self.a ** 2 / self.h + self.h / self.tau - self.c * self.h + (self.delta / self.gamma) * (2 * self.a ** 2 - self.b * self.h)
                dn = (self.h / self.tau) * self.U[i, self.N] + self._phi1 * (2 * self.a ** 2 + self.b * self.h) / self.gamma

                bound2 = (dn - an * self.U[i + 1, self.N - 1]) / bn

        self.U[i + 1, 0] = bound1
        self.U[i + 1, self.N] = bound2

    def implicit(self, i):
        self.resetTmp()

        for j in range(1, self.N):
            res = 0.0
            res += (1.0 - self.e * self.tau / 2.0) * self.U[i - 1, j]
            res -= 2.0 * self.U[i, j]
            res -= self.tau ** 2 * self.execEquationFunction(self.vecX[j], self.tNext)

            self.mat[j, j - 1] = self.A
            self.mat[j, j] = self.B
            self.mat[j, j + 1] = self.C
            self.vec[j] = res

        if self.boundCondType == '2_1':
            self.mat[0, 0] = self.beta - self.alpha / self.h
            self.mat[0, 1] = self.alpha / self.h
            self.mat[self.N, self.N - 1] = -self.gamma / self.h
            self.mat[self.N, self.N] = self.delta + self.gamma / self.h
            self.vec[0] = self._phi0
            self.vec[self.N] = self._phi1

        elif self.boundCondType == '3_2':
            h2 = 2.0 * self.h
            self.mat[0, 0] = self.beta - 3.0 * self.alpha / h2
            self.mat[0, 1] = 4.0 * self.alpha / h2
            self.mat[0, 2] = -self.alpha / h2
            self.mat[self.N, self.N - 2] = self.gamma / h2
            self.mat[self.N, self.N - 1] = -4.0 * self.gamma / h2
            self.mat[self.N, self.N] = self.delta + 3.0 * self.gamma / h2
            self.vec[0] = self._phi0
            self.vec[self.N] = self._phi1

        elif self.boundCondType == '2_12':
            if self.alpha == 0:
                self.mat[0, 0] = self.beta
                self.vec[0] = self._phi0
            else:
                b0 = 2.0 * self.a ** 2 / self.h + self.h / self.tau - self.c * self.h - (self.beta / self.alpha) * (2.0 * self.a ** 2 - self.b * self.h)
                c0 = -2.0 * self.a ** 2 / self.h
                d0 = (self.h / self.tau) * self.U[i, 0] - self._phi0 * (2.0 * self.a **2 - self.b * self.h) / self.alpha

                self.mat[0, 0] = b0
                self.mat[0, 1] = c0
                self.vec[0] = d0

            if self.gamma == 0:
                self.mat[self.N, self.N] = self.delta
                self.vec[self.N] = self._phi0
            else:
                an = -2.0 * self.a ** 2 / self.h
                bn = 2.0 * self.a ** 2 / self.h + self.h / self.tau - self.c * self.h + (self.delta / self.gamma) * (2.0 * self.a ** 2 + self.b * self.h)
                dn = (self.h / self.tau) * self.U[i, self.N] + self._phi1 * (2.0 * self.a ** 2 + self.b * self.h) / self.gamma

                self.mat[self.N, self.N - 1] = an
                self.mat[self.N, self.N] = bn
                self.vec[self.N] = dn

        self.vecRes = np.linalg.solve(self.mat, self.vec)

        for j in range(self.N + 1):
            self.U[i + 1, j] = self.vecRes[j]

    def createTmp(self):
        self.mat = np.ndarray((self.N + 1, self.N + 1))
        self.vec = np.ndarray((self.N + 1))
        self.vecRes = np.ndarray((self.N + 1))

    def resetTmp(self):
        self.mat.fill(0)
        self.vec.fill(0)
        self.vecRes.fill(0)

    def setCoefsSLE(self):
        self.setSigmas()
        self.A = self.sigma1 - self.sigma3
        self.B = self.sigma2 - 2.0 * self.sigma1
        self.C = self.sigma1 + self.sigma3

    def setSigmas(self):
        self.sigma1 = (self.tau ** 2) * (self.a ** 2) / (self.h ** 2)
        self.sigma2 = (self.tau ** 2) * self.c - 1.0 - self.e * self.tau / 2.0
        self.sigma3 = (self.tau ** 2) * self.b / (2.0 * self.h)

    def setInitialVecX(self):
        for j in range(self.N + 1):
            self.U[0, j] = self.execFirstInitial(self.vecX[j])
            if self.initCondType == '1':
                self.U[1, j] = self.U[0, j] + self.execSecondInitial(self.vecX[j]) * self.tau
            elif self.initCondType == '2':
                self.U[1, j] += self.a ** 2 * self.execSecondInitialDerivative(self.vecX[j])
                self.U[1, j] += self.b * self.execFirstInitialDerivative(self.vecX[j])
                self.U[1, j] += self.c * self.U[0, j]
                self.U[1, j] += self.execEquationFunction(self.vecX[j], self.vecT[1])
                self.U[1, j] *= self.tau ** 2 / 2
                self.U[1, j] += self.execSecondInitial(self.vecX[j]) * self.tau
                self.U[1, j] += self.U[0, j]

    def execFirstInitial(self, x):
        return self.mathParser.parse(self.psi1).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            'x': x
        })

    def execSecondInitial(self, x):
        return self.mathParser.parse(self.psi2).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            'x': x
        })

    def execFirstInitialDerivative(self, x):
        return self.mathParser.parse(self.psi1Derivative1).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            'x': x
        })

    def execSecondInitialDerivative(self, x):
        return self.mathParser.parse(self.psi1Derivative2).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            'x': x
        })

    def execLeftBound(self, t):
        return self.mathParser.parse(self.phi0).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            't': t
        })

    def execRightBound(self, t):
        return self.mathParser.parse(self.phiN).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            't': t
        })

    def execEquationFunction(self, x, t):
        return self.mathParser.parse(self.f).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            'x': x,
            't': t
        })


    def calcReferenceGrid(self):
        for i in range(self.K + 1):
            for j in range(self.N + 1):
                self.referenceGrid[i, j] = self.execReference(self.vecX[j], self.vecT[i])

    def execReference(self, x, t):
        return self.mathParser.parse(self.reference).evaluate({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'e': self.e,
            'x': x,
            't': t,
        })

if __name__ == "__main__":
    #var10
    args = {
        'a': 1,
        'b': 1,
        'c': -1,
        'e': 3,
        'l': 3.14,
        'alpha': 1,
        'beta': 0,
        'gamma': 1,
        'delta': 0,
        'T': 5,
        'K': 100,
        'N': 10,
        'f': '-cos(x) * exp(-t)',
        'psi1': 'sin(x)',
        'psi2': '-sin(x)',
        'phi0': 'exp(-t)',
        'phiN': '-exp(-t)',
        'reference': 'exp(-t) * sin(x)',
        'psi1Derivative1': 'cos(x)',
        'psi1Derivative2': '-sin(x)',
        'boundCondType': '3_2',
        'schemeType': 'IMPLICIT',
        'initCondType': 'INITIAL_CONDITION_1'
    }
    plt.imshow(HyperbolicSolver(args).getResultsDict()['U'])
    plt.show() 