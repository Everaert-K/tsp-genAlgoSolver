import Reporter
import numpy as np
import random
import math

## --------------- Initialization --------------- ##
def initialize(population_size,number_of_nodes):
    population = []
    for i in range(population_size):
        individual = np.arange(number_of_nodes)
        np.random.shuffle(individual)
        population.append(individual)
    population = np.asarray(population)
    return population


## --------------- Recombination --------------- ##
def PMX(parent1, parent2):

    child1 = parent1.copy()
    child2 = parent2.copy()
    size = min(len(child1), len(child2))
    p1, p2 = [0] * size, [0] * size
    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[child1[i]] = i
        p2[child2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = child1[i]
        temp2 = child2[i]
        # Swap the matched value
        child1[i], child1[p1[temp2]] = temp2, temp1
        child2[i], child2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
    return child1, child2

def PMX2(parent1, parent2, length):
    child1 = parent1.copy()
    child2 = parent2.copy()
    size = min(len(child1), len(child2))
    p1, p2 = [0] * size, [0] * size
    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[child1[i]] = i
        p2[child2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size-length)
    # cxpoint2 = random.randint(0, size - 1)
    cxpoint2 = cxpoint1 + length
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = child1[i]
        temp2 = child2[i]
        # Swap the matched value
        child1[i], child1[p1[temp2]] = temp2, temp1
        child2[i], child2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
    return child1, child2


# calculates the length of a path
def length(individual: np.array, distance_matrix: np.array) -> float:
    distance = 0
    size = distance_matrix.shape[0]
    for i in range(size - 1):
        distance += distance_matrix[individual[i]][individual[i + 1]]
    distance += distance_matrix[individual[-1]][individual[0]]
    return distance

## --------------- Elimination --------------- ##
def elimination(population: np.array, offspring: np.array, distance_matrix: np.array, population_size: int) -> np.array:
    combined = np.concatenate((population, offspring), axis=0)
    combined = list(combined.astype(int))
    combined.sort(key=lambda individual: length(individual, distance_matrix))
    new_population = combined[:population_size]
    return np.array(new_population)

## --------------- Mutation --------------- ##
def mutation(individual: np.array, alpha: float) -> np.array:
    '''
    swaps 2 subsets where the size of the subset is an integer size determined by alpha between 1 and len(individual)/2

    :param individual: numpy array, containing the order of visiting the locations
    :param alpha: float, chance to increase subset size by 1 each iteration
    :return:
    '''
    subset_size = 0

    individual_size = len(individual)

    # double size of individual since we are want to treat it as a continuous path without beginning or end
    individual = np.append(individual, individual)

    # determine size of individual maybe making a pdf could be better
    while random.random() < alpha:
        subset_size += 1
        if subset_size == math.floor(individual_size / 2):
            break

    first_index = random.randrange(individual_size)

    # determine position of swapping subset so it can't overlap
    offset = first_index + subset_size + random.randrange(individual_size - 2 * subset_size)

    temp = individual[first_index:first_index + subset_size].copy()
    individual[first_index:first_index + subset_size] = individual[offset:offset + subset_size]
    individual[offset:offset + subset_size] = temp

    '''
    possible alternative:
    while random.random() < alpha:
        first_index = random.randrange(individual_size)
        second_index = random.randrange(individual_size)
        temp = individual[first_index]
        individual[first_index] = individual[second_index]
        individual[second_index] = temp
    return individual
    '''

    return individual[first_index:first_index + individual_size]

## --------------- Selection --------------- ##
def selection(population: np.array, k: int, distance_matrix: np.array):
    '''
    k tournament selection, selects k random individual then selects the best from that group
    :param population: numpy array, containing all the individuals
    :param k: int, initial size of random selected group
    :param distance_matrix: numpy array, necessary for fitness calculation
    :return:
    '''
    random_selection = random.choices(population, k=k)
    random_selection.sort(key=lambda individual: length(individual, distance_matrix))
    return random_selection[0]


# Modify the class name to match your student number.
class r0856880:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename, population_size, its, recom_its, k,alpha):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        population = initialize(population_size,distanceMatrix.shape[0])
        its_start = its
        i = 0

        while (its > i):

            # Your code here.
            offspring = np.zeros([2*recom_its,distanceMatrix.shape[0]])
            # Recombination
            for j in range(0,2*recom_its,2):
                # NIEUW
                # offset = (its_start - its) * k / its_start
                parent1 = selection(population,k, distanceMatrix)
                parent2 = selection(population,k, distanceMatrix)

                child1,child2 = PMX(parent1,parent2)
                # child1, child2 = PMX2(parent1, parent2, 2)
                offspring[j] = child1
                offspring[j+1] = child2

            # Mutation
            for j in range(len(offspring)):
                # offspring[j] =mutation(offspring[j],alpha)
                # NIEUW
                offset = (its_start-its)*alpha/its_start
                offspring[j] = mutation(offspring[j], alpha-offset)

            for j in range(len(population)):
                # NIEUW
                offset = (its_start - its) * alpha / its_start
                population[j] =mutation(population[j],alpha-offset)

            # Elimination
            population = elimination(population,offspring,distanceMatrix,population_size)
            print("Score iteration {}".format(i),length(population[0],distanceMatrix))
            
            bestSolution = population[0]
            bestObjective = length(bestSolution, distanceMatrix)
            meanObjective = np.average([length(individual, distanceMatrix) for individual in population])
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            # timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            # if timeLeft < 0:
            #    break
            i+=1

        return 0


TSP = r0856880()
TSP.optimize("tour29.csv",50,500,25,10,0.5)

# parent1 = np.array([1, 5, 3, 2, 4])
# parent2 = np.array([2, 3, 1, 4, 5])
# print(parent1)
# PMX(parent1,parent2)
#child1, child2 = PMX(parent1,parent2)
#print(child1)
#print(child2)












