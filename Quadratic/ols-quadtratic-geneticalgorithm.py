class Equation(object):
    quadtratic_c = 0
    linear_c = 0
    constant = 0
    fitness = 0
    nfitness = 0

    def __init__(self, quadtratic, linear, constant):
        self.quadtratic_c = quadtratic
        self.linear_c = linear
        self.constant = constant
        self.fitness = 0
        self.nfitness = 0

    def calculateLineY(self, x):
        return self.constant + self.linear_c * x + self.quadtratic_c * x * x

    def calculateDistance(self, point):
        distance = point.y - self.calculateLineY(point.x)
        return distance

    def calculateFitness(self, points_array):
        for point in points_array:
            self.fitness += (self.calculateDistance(point) ** 2) **3
        self.fitness = 1 / self.fitness

    def mutate(self, rate):
        if ranFloat(0, 1) < rate:
            self.quadtratic_c = ranFloat(MAXIMUMQUADRATIC, MINIMUMQUADRATIC)
            self.linear_c = ranFloat(MAXIMUMLINEAR, MINIMUMLINEAR)
            self.constant = ranFloat(MAXIMUMCONSTANT, MINIMUMCONSTANT)

class Point(object):
    x = 0.0
    y = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

import random, numpy, collections

def readData(location):
    return open(location).read()

def returnDataList(data, delimiter):
    return list(filter(None, data.split(delimiter)))

def convertToFloat(data):
    return list(map(float, data))

def convertToPoints(x_array, y_array):
    points_array = []
    for x, y in zip(x_array, y_array):
        points_array.append(Point(x, y))
    return points_array

def returnPoints():
    X = convertToFloat(returnDataList(readData('angle.txt'), '\n'))
    Y = convertToFloat(returnDataList(readData('distance.txt'), '\n'))
    return convertToPoints(X, Y)

def ranFloat(maximum, minimum):
    return random.uniform(maximum, minimum)

def generateEquations(num, maxL, minL, maxQ, minQ, maxC, minC):
    population = []
    for n in range(num):
        quadtratic_c = ranFloat(maxQ, minQ)
        print(quadtratic_c)
        linear_c = ranFloat(maxL, minL)
        constant = ranFloat(maxC, minC)
        population.append(Equation(quadtratic_c, linear_c, constant))
    return population

def assessPopulation(population, points_array):
    for equation in population:
        equation.calculateFitness(points_array)

def normalizeFitness(population):
    total_fitness = 0
    for equation in population:
        total_fitness += equation.fitness
    for equation in population:
        equation.nfitness = equation.fitness / total_fitness

def pickByFitness(population, nfitness):
    return numpy.random.choice(population, p=nfitness)

def chooseParents(population):
    Pair = collections.namedtuple('Point', ['p1', 'p2'])
    parents = []
    nfitness = []
    for equation in population:
        nfitness.append(equation.nfitness)
    for i in range(0, len(population)):
        p1 = pickByFitness(population, nfitness)
        p2 = pickByFitness(population, nfitness)
        parents.append(Pair(p1, p2))
    return parents

def average(value1, value2):
    return (value1 + value2) / 2

def breed(parents):
    npopulation = []
    for pair in parents:
        nquad = average(pair.p1.quadtratic_c, pair.p2.quadtratic_c)
        nlin = average(pair.p1.linear_c, pair.p2.linear_c)
        ncon = average(pair.p1.constant, pair.p2.constant)
        child = Equation(nquad, nlin, ncon)
        child.mutate(MUTATION_RATE)
        npopulation.append(child)
    return npopulation

def pickBest(population):
    best_equation = population[0]
    for equation in population:
        if best_equation.fitness <= equation.fitness:
            best_equation = equation
    return best_equation

def averageFitness(population):
    total = 0
    for equation in population:
        total += equation.fitness
    return total/len(population)

def drawScatterGraph(points_array, formula, title):
    import matplotlib.pyplot as plt
    x_array = []
    y_array = []
    for point in points_array:
        x_array.append(point.x)
        y_array.append(point.y)
    x = numpy.linspace(1, 100, 500)
    plt.scatter(x_array, y_array)
    plt.plot(x, formula.calculateLineY(x))
    plt.show()

def output(bE, fE, af, generation):
        print("Generation ===> {}".format(generation))
        print("Best Equation ===> y = {}".format(bE))
        print("Best Fitness ===> {}".format(fE.fitness))
        print("Average Fitness ===> {}".format(af))
        print("\n\n")

def evolve(generations, population):
    fE = None
    for generation in range(generations):
        assessPopulation(population, points_array)
        fE = pickBest(population)
        af = averageFitness(population)
        bE = "{}*x^2 + {}*x + {}".format(fE.quadtratic_c, fE.linear_c, fE.constant)
        normalizeFitness(population)
        parents = chooseParents(population)
        population = breed(parents)
        output(bE, fE, af, generation)
    return fE

MAXIMUMQUADRATIC =  +5
MINIMUMQUADRATIC =  -5
MAXIMUMLINEAR    =  +15
MINIMUMLINEAR    =  -15
MAXIMUMCONSTANT  =  +25
MINIMUMCONSTANT  =  -25
POPULATIONSIZE   =  +200
MUTATION_RATE    =  +0.9
NUM_GENERATIONS  =  +100

points_array = returnPoints()
population = generateEquations(POPULATIONSIZE, MAXIMUMLINEAR, MINIMUMLINEAR, MAXIMUMQUADRATIC, MINIMUMQUADRATIC, MAXIMUMCONSTANT, MINIMUMCONSTANT)
bestEquation = evolve(NUM_GENERATIONS, population)
drawScatterGraph(points_array, bestEquation,  "quadtratic regressional analysis with Genetic Algorithm")
