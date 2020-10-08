import math
import numpy as np
import matplotlib.pyplot as plt
from py_expression_eval import Parser
from pprint import pprint

class Solver(object):

    def __init__(self, args):
        pprint(args)

        self.mathParser = Parser()
        self.parseArgs(args)
        self.createGrids()
        self.createVectors()
        self.setGridSteps()
        self.setReference(args)

    def createGrids(self):
        self.createFunctionGrid()
        self.createTmpGrid()

    def createFunctionGrid(self):
        self.U = np.ndarray((self.ny + 1, self.nx + 1))
        self.U.fill(0)

    def createTmpGrid(self):
        self.tmpGrid = np.ndarray((self.ny + 1, self.nx + 1))
        self.tmpGrid.fill(0)

    def createVectors(self):
        self.createVectorX()
        self.createVectorY()

    def createVectorX(self):
        self.vecX = np.linspace(0, self.lx, self.nx + 1)

    def createVectorY(self):
        self.vecY = np.linspace(0, self.ly, self.ny + 1)

    def parseArgs(self, args):
        self.parseEquation(args)
        self.parseBound(args)

    def parseEquation(self, args):
        self.parseEquationCoefs(args)
        self.parseEquationFunction(args)
        self.parseEquationParams(args)

    def parseEquationParams(self, args):
        self.omega = args['omega']
        self.eps = args['eps']
        self.methodType = args['methodType']
        self.maxIterations = args['maxIterations']
        self.parseParamsY(args)
        self.parseParamsX(args)

    def parseEquationFunction(self, args):
        self.f = args['f']

    def parseEquationCoefs(self, args):
        self.bx = args['bx']
        self.by = args['by']
        self.c = args['c']

    def parseParamsX(self, args):
        self.lx = args['lx']
        self.nx = args['nx']

    def parseParamsY(self, args):
        self.ly = args['ly']
        self.ny = args['ny']

    def parseBound(self, args):
        self.parseLeftBound(args)
        self.parseRightBound(args)
        self.parseTopBound(args)
        self.parseDownBound(args)

    def parseLeftBound(self, args):
        self.alpha1 = args['alpha1']
        self.beta1 = args['beta1']
        self.phi1 = args['phi1']

    def parseRightBound(self, args):
        self.alpha2 = args['alpha2']
        self.beta2 = args['beta2']
        self.phi2 = args['phi2']

    def parseTopBound(self, args):
        self.alpha3 = args['alpha3']
        self.beta3 = args['beta3']
        self.phi3 = args['phi3']

    def parseDownBound(self, args):
        self.alpha4 = args['alpha4']
        self.beta4 = args['beta4']
        self.phi4 = args['phi4']

    def setGridSteps(self):
        self.setStepX()
        self.setStepY()

    def setStepX(self):
        self.hx = self.lx / self.nx

    def setStepY(self):
        self.hy = self.ly / self.ny

    def getResultsDict(self):
        self.solve()
        res =  {
            'x': self.vecX,
            'y': self.vecY,
            'iterations': self.iterations,
            'U': self.U
        }
        if self.reference != '':
            self.calcReferenceGrid()
            res['exact_solution'] = self.referenceGrid
        return res

    def solve(self):
        self.iterations = 0
        self.error = self.eps + 1.0

        while (self.error > self.eps):
            for i in range(1, self.ny):
                for j in range(1, self.nx):
                    res = 0.0
                    if self.methodType == 'LIEBMANN':
                        res = self.Liebmann(i, j)
                    elif self.methodType == 'SEIDEL':
                        res = self.Seidel(i, j)

                    res += self.execEquationFunction(i, j)
                    res /= 2.0 / (self.hx ** 2) + 2.0 / (self.hy ** 2) - self.c
                    res = (1 - self.omega) * self.U[i, j] + res * self.omega

                    self.tmpGrid[i, j] = res
                    self.tmpGrid[0, j] = self.execTopBound(self.tmpGrid, j)
                    self.tmpGrid[self.ny, j] = self.execDownBound(self.tmpGrid, j)

                    self.calcErrorX(i, j)
                self.calcErrorY(i, j)

                self.tmpGrid[i, 0] = self.execLeftBound(self.tmpGrid, i)
                self.tmpGrid[i, self.nx] = self.execRightBound(self.tmpGrid, i)

            self.U = self.tmpGrid.copy()
            self.iterations+=1
            if self.iterations > self.maxIterations:
                break
        self.calcCornersValue()

    def Liebmann(self, i, j):
        res = 0.0
        res += (self.U[i, j + 1] + self.U[i, j - 1]) / (self.hx ** 2)
        res += (self.U[i + 1, j] + self.U[i - 1, j]) / (self.hy ** 2)
        res += self.bx * (self.U[i, j + 1] - self.U[i, j - 1]) / (2.0 * self.hx)
        res += self.by * (self.U[i + 1, j] - self.U[i - 1, j]) / (2.0 * self.hy)
        return res

    def Seidel(self, i, j):
        res = 0.0
        res += (self.U[i, j + 1] + self.tmpGrid[i, j - 1]) / (self.hx ** 2)
        res += (self.U[i + 1, j] + self.tmpGrid[i - 1, j]) / (self.hy ** 2)
        res += self.bx * (self.U[i, j + 1] - self.tmpGrid[i, j - 1]) / (2.0 * self.hx)
        res += self.by * (self.U[i + 1, j] - self.tmpGrid[i - 1, j]) / (2.0 * self.hy)
        return res

    def execEquationFunction(self, i, j):
        return self.mathParser.parse(self.f).evaluate({
            'x': self.vecX[j],
            'y': self.vecY[i]
        })

    def execLeftBound(self, mat, i):
        res = self.mathParser.parse(self.phi1).evaluate(
            {
                'y': self.vecY[i]
            })
        res -= mat[i, 1] * self.alpha1 / self.hx
        res /= (self.beta1 - self.alpha1 / self.hx)
        return res

    def execRightBound(self, mat, i):
        res = self.mathParser.parse(self.phi2).evaluate(
            {
                'y': self.vecY[i]
            })
        res += mat[i, self.nx - 1] * self.alpha2 / self.hx
        res /= (self.beta2 + self.alpha2 / self.hx)
        return res

    def execTopBound(self, mat, j):
        res = self.mathParser.parse(self.phi3).evaluate(
            {
                'x': self.vecX[j]
            })
        res -= mat[1, j] * self.alpha3 / self.hy
        res /= (self.beta3 - self.alpha3 / self.hy)
        return res

    def execDownBound(self, mat, j):
        res = self.mathParser.parse(self.phi4).evaluate(
            {
                'x': self.vecX[j]
            })
        res += mat[self.ny - 1, j] * self.alpha4 / self.hy
        res /= (self.beta4 + self.alpha4 / self.hy)
        return res

    def calcErrorX(self, i, j):
        self.error = max(self.error, abs(self.tmpGrid[i, j] - self.U[i, j]))
        self.error = max(self.error, abs(self.tmpGrid[0, j] - self.U[0, j]))
        self.error = max(self.error, abs(self.tmpGrid[self.ny, j] - self.U[self.ny, j]))

    def calcErrorY(self, i, j):
        self.error = max(self.error, abs(self.tmpGrid[i, 0] - self.U[i, 0]))
        self.error = max(self.error, abs(self.tmpGrid[i, self.nx] - self.U[i, self.nx]))

    def calcCornersValue(self):
        self.U[0, 0] = self.execLeftBound(self.U, 0)
        self.U[0, self.nx] = self.execRightBound(self.U, 0)
        self.U[self.ny, 0] = self.execLeftBound(self.U, self.ny)
        self.U[self.ny, self.nx] = self.execLeftBound(self.U, self.ny)

    def setReference(self, args):
        if not 'reference' in args:
            self.reference = ''
            return
        self.reference = args['reference']
        self.referenceGrid = np.ndarray((self.ny + 1, self.nx + 1))
        self.referenceGrid.fill(0)

    def calcReferenceGrid(self):
        for i in range(self.ny + 1):
            for j in range(self.nx + 1):
                self.referenceGrid[i, j] = self.execReference(i, j)
        return self.referenceGrid

    def execReference(self, i, j):
        return self.mathParser.parse(self.reference).evaluate({
            'x': self.vecX[j],
            'y': self.vecY[i]
        })

if __name__ == "__main__":
    args = {
        'bx': 0,
        'by': 0,
        'c': 0,
        'lx': 1,
        'ly': 1,
        'omega': 0.5,
        'eps': 0.01,
        'maxIterations': 100,
        'alpha1': 0,
        'alpha2': 0,
        'alpha3': 0,
        'alpha4': 0,
        'beta1': 1,
        'beta2': 1,
        'beta3': 1,
        'beta4': 1,
        'nx': 10,
        'ny': 10,
        'f': '0',
        'phi1': 'y',
        'phi2': '1 + y',
        'phi3': 'x',
        'phi4': '1 + x',
        'reference': 'x + y',
        'methodType': 'LIEBMANN'
    }

    plt.imshow(Solver(args).getResultsDict()['U'])
    plt.show()