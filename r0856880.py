import Reporter
import numpy as np
import random
import math

import sys
import time

## --------------- Initialization --------------- ##
def initialize(population_size,number_of_nodes):

    '''
    initializes a random population of size population_size where each individual contains number_of_nodes nodes

    :param population_size: integer, determines the number of individual inside the population
    :param number_of_nodes: integer, the number of nodes out of which each individual will exist
    :return: population, numpy array
    '''

    population = []
    for i in range(population_size):
        individual = np.arange(number_of_nodes)
        np.random.shuffle(individual)
        population.append(individual)
    population = np.asarray(population)
    return population

# NIEUW
def local_search(individual, k, depth, distanceMatrix):
    number_of_nodes = len(individual)
    individual_copy = individual.copy()
    for d in range(depth):
        temp_indiv = individual_copy.copy()
        for j in range(k):
            possible_better_candidate = temp_indiv.copy()
            index1 = random.randint(0, number_of_nodes - 1)
            index2 = random.randint(0, number_of_nodes - 1)
            while (index1 == index2):
                index2 = random.randint(0, number_of_nodes - 1)
            possible_better_candidate[index1], possible_better_candidate[index2] = possible_better_candidate[index2], possible_better_candidate[index1]
            if length(possible_better_candidate, distanceMatrix) < length(individual_copy, distanceMatrix):
                individual_copy = possible_better_candidate.copy()
    return individual_copy

# NIEUW
# Now includes some local search in order to start out with a better starting population
def initialize2(population_size,number_of_nodes, alpha, k, distanceMatrix):

    '''
    initializes a random population of size population_size where each individual contains number_of_nodes nodes, in addition to this
    it performes some local search on certain idividuals in order to start out with a better initial population

    :param population_size: integer, determines the number of individual inside the population
    :param number_of_nodes: integer, the number of nodes out of which each individual will exist
    :param alpha: float, determines the chance for which local search will be applied to an individual
    :param k: integer, the number of random variations to which an individual will be compared during the local search
    :param distanceMatrix: numpy array, a numpy array of numpy arrays that contains all the distances between the different nodes
    :return: population, numpy array
    '''

    population = []
    for i in range(population_size):
        individual = np.arange(number_of_nodes)
        np.random.shuffle(individual)
        if random.random() < alpha:
            # if you come here then local search will be performes
            #for j in range(k):
            #    possible_better_candidate = individual.copy()
            #    index1 = random.randint(0, number_of_nodes-1)
            #    index2 = random.randint(0, number_of_nodes-1)
            #    while(index1 == index2):
            #        index2 = random.randint(0, number_of_nodes - 1)
            #    possible_better_candidate[index1], possible_better_candidate[index2] = possible_better_candidate[index2], possible_better_candidate[index1]
            #    if length(possible_better_candidate, distanceMatrix) < length(individual, distanceMatrix):
            #        individual = possible_better_candidate.copy()
            individual = local_search(individual, k, 3,distanceMatrix) # 3 determines the depth of local search
        population.append(individual)
    population = np.asarray(population)
    return population


## --------------- Recombination --------------- ##
def PMX(parent1, parent2):

    '''
    Performs the recombination step of the algorithm, it swaps all points between 2 cutpoints and then solves inconsistencies

    :param parent1: numpy array, The first individual
    :param parent2: numpy array, the second individual
    :return: child1, child2: numpy array, the 2 generated children
    '''

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

def PMX2(parent1, parent2, alpha):

    '''
    Performs the recombination step of the algorithm, it swaps all points between 2 cutpoints and then solves inconsistencies
    it uses an extra parameter that determines the maximum length of the distance between the 2 cutpoints

    :param parent1: numpy array, The first individual
    :param parent2: numpy array, the second individual
    :param alpha: float, indicates what percentage of the size can be swapped between the 2 parents
    :return: child1, child2: numpy array, the 2 generated children
    '''

    child1 = parent1.copy()
    child2 = parent2.copy()
    size = len(child1)

    # determines the maximum number of places that can be between cxpoint1 and cxpoint2
    # alpha starts as 1
    max_between = (size-1)*alpha

    p1, p2 = [0] * size, [0] * size
    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[child1[i]] = i
        p2[child2[i]] = i
    # Choose crossover points

    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)

    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1 # ???
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    # Apply crossover between cx points

    if (cxpoint2-cxpoint1-1)>max_between:
        cxpoint2 = cxpoint1 + max_between + 1

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

    '''
    returns the length of an individual

    :param individual: numpy array
    :param distance_matrix: numpy array, a numpy array of numpy arrays that contains all the distances between the different nodes
    :return: distance, integer
    '''

    distance = 0
    size = distance_matrix.shape[0]
    for i in range(size - 1):
        distance += distance_matrix[int(individual[i])][int(individual[i + 1])]
    distance += distance_matrix[int(individual[-1])][int(individual[0])]
    return distance

## --------------- Elimination --------------- ##
def elimination(population: np.array, offspring: np.array, distance_matrix: np.array, population_size: int) -> np.array:

    '''
    The elimination step only keep the (population_size) best individuals for the new population

    :param population: numpy array, contains the individuals of the current population
    :param offspring: numpy array, contains the individuals generated by recombination of the current population
    :param distance_matrix: numpy array, a numpy array of numpy arrays that contains all the distances between the different nodes
    :param population_size: integer, represents the number of individuals that a population contains
    :return: the new population, numpy array
    '''

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

        # if timeLeft >= 150 then you should take 10 best and start doing less mutation in order to recombine
        timeLeft = 300.0 # initialize on 5 minutes

        # population = initialize(population_size,distanceMatrix.shape[0])
        population = initialize2(population_size,distanceMatrix.shape[0], 0.5, 50, distanceMatrix)
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

                child1,child2 = PMX2(parent1,parent2, 1-((its_start-its)/its_start))
                offspring[j] = child1
                offspring[j+1] = child2

            chance_local_search = 0.5
            generated_by_local_search = 1

            # Mutation
            for j in range(len(offspring)):
                # NIEUW
                offset = (its_start-its)*alpha/its_start
                # NIEUW
                # perform local search after variation
                mutated = mutation(offspring[j], alpha-offset)
                if random.random() < chance_local_search:
                    for z in range(generated_by_local_search):
                        possible_better_candidate = mutated.copy()
                        index1 = random.randint(0, distanceMatrix.shape[0] - 1)
                        index2 = random.randint(0, distanceMatrix.shape[0] - 1)
                        while index1 == index2:
                            index2 = random.randint(0, distanceMatrix.shape[0] - 1)
                        possible_better_candidate[index1], possible_better_candidate[index2] = possible_better_candidate[index2], possible_better_candidate[index1]
                        if length(possible_better_candidate, distanceMatrix) < length(mutated, distanceMatrix):
                            mutated = possible_better_candidate.copy()
                offspring[j] = mutated

            # don't mutate the best topPercent of the previous population
            sorted(population, key=lambda individual: length(individual, distanceMatrix))
            if timeLeft <= 150.0:
                # startIndex = int(0.1*population_size)
                startIndex = int(0.05 * population_size)
            else:
                if length(population[0],distanceMatrix) == np.inf:
                    startIndex = max(2,int(0.005 * population_size))
                else:
                    startIndex = int(0.05 * population_size)
            for j in range(startIndex, len(population)):
                # NIEUW
                offset = (its_start - its) * alpha / its_start
                # NIEUW
                # perform local search after variation
                mutated = mutation(population[j], alpha - offset)
                if random.random() < chance_local_search:
                    for z in range(generated_by_local_search):
                        possible_better_candidate = mutated.copy()
                        index1 = random.randint(0, distanceMatrix.shape[0] - 1)
                        index2 = random.randint(0, distanceMatrix.shape[0] - 1)
                        while (index1 == index2):
                            index2 = random.randint(0, distanceMatrix.shape[0] - 1)
                        possible_better_candidate[index1], possible_better_candidate[index2] = possible_better_candidate[index2], possible_better_candidate[index1]
                        if length(possible_better_candidate, distanceMatrix) < length(mutated, distanceMatrix):
                            mutated = possible_better_candidate.copy()
                population[j] = mutated


            # Elimination
            population = elimination(population,offspring,distanceMatrix,population_size)
            print("Score iteration {}".format(i),length(population[0],distanceMatrix))
            
            bestSolution = population[0]
            bestObjective = length(bestSolution, distanceMatrix)
            meanObjective = np.average([length(individual, distanceMatrix) for individual in population])

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
            i+=1

        return 0


TSP = r0856880()
# def optimize(filename, population_size, its, recom_its, k,alpha):
TSP.optimize("tour29.csv",75,5000000,50,10,0.5)




















