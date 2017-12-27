class Equation(object):
    gradient = 0
    intercept = 0
    fitness = 0
    nfitness = 0

    def __init__(self, gradient, intercept):
        self.gradient = gradient
        self.intercept = intercept
        self.fitness = 0
        self.nfitness = 0

    def calculateDistance(self, point):
        distance = point.y - ( self.gradient * point.x ) + self.intercept
        return distance

    def calculateFitness(self, points_array):
        for point in points_array:
            self.fitness += self.calculateDistance(point) ** 2
        self.fitness = (1 / self.fitness)**3

    def mutate(self, rate):
        if ranFloat(0, 1) < rate:
            self.gradient += ranFloat(MAXIMUMGRADIENT, MINIMUMGRADIENT)
            self.intercept += ranFloat(MAXIMUMINTERCEPT, MINIMUMINTERCEPT)

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
    X = convertToFloat(returnDataList(readData('bodyweights.txt'), '\n'))
    Y = convertToFloat(returnDataList(readData('brainweights.txt'), '\n'))
    return convertToPoints(X, Y)

def ranFloat(maximum, minimum):
    return random.uniform(maximum, minimum)

def generateEquations(num, maxI, minI, maxG, minG):
    population = []
    for n in range(num):
        gradient = ranFloat(maxG, minG)
        intercept = ranFloat(maxI, minI)
        population.append(Equation(gradient, intercept))
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
        ngradient = average(pair.p1.gradient, pair.p2.gradient)
        nintercept = average(pair.p1.intercept, pair.p2.intercept)
        child = Equation(ngradient, nintercept)
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

    x = numpy.array([0, 5000])
    f = lambda x: (formula.gradient * x) + formula.intercept
    plt.plot(x, f(x))
    plt.scatter(y_array, x_array)
    plt.title(title)
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
        bE = "{}*x + {}".format(fE.gradient, fE.intercept)
        normalizeFitness(population)
        parents = chooseParents(population)
        population = breed(parents)
        output(bE, fE, af, generation)
    return fE

MAXIMUMGRADIENT  =  +20
MINIMUMGRADIENT  =  -20
MAXIMUMINTERCEPT =  +60
MINIMUMINTERCEPT =  -60
POPULATIONSIZE   =  +500
MUTATION_RATE    =  +0.5
NUM_GENERATIONS  =  +100

points_array = returnPoints()
population = generateEquations(POPULATIONSIZE,  MAXIMUMINTERCEPT, MINIMUMINTERCEPT, MAXIMUMGRADIENT, MINIMUMGRADIENT)
bestEquation = evolve(NUM_GENERATIONS, population)
drawScatterGraph(points_array, bestEquation,  "regressional analysis with Genetic Algorithm")
