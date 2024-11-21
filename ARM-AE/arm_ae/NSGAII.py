#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import copy
import random
import json

import numpy
import pandas
import pandas as pd
from math import sqrt
import time

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from BenchmarkManager import BenchmarkManager

datasetName = 'plants'
data = pd.read_csv('../data/'+datasetName+'.csv',index_col=0,dtype=float,header=0)
dataSize = len(data.loc[0])



def initIndividual(nb,LIMIT_ANTECEDENT,LIMIT_CONSEQUENT):
    antecedents = numpy.random.randint(int(nb/2),size = LIMIT_ANTECEDENT)
    consequents = numpy.random.randint( int(nb/2), size=LIMIT_CONSEQUENT)
    rule = numpy.zeros(nb,dtype=int)
    rule[antecedents] = 1
    rule[consequents] = 1
    positionAntecedents = antecedents + (int(nb/2))
    positionConsequents = consequents + (int(nb/2))
    rule[positionAntecedents] = 1
    rule[positionConsequents] = 0
    return list(rule)

def computeMeasures(individual,data):
    data = data.to_numpy()
    cpIndividual = numpy.array(individual)
    dataSize = len(data)
    delimitation = int(len(cpIndividual)/2)
    position = cpIndividual[delimitation:] == 1
    rules = cpIndividual[:delimitation] == 1
    positionIndex = []
    for ind in range(len(rules)):
        if rules[ind] == 1 and position[ind] == 1:
            positionIndex.append(ind)
    positionIndex = numpy.array(positionIndex)
    rows = data[:,rules]
    PA = 0
    PAC = 0
    for row in rows:
        if sum(row) == sum(rules):
            PAC+=1
    PAC/=dataSize
    if len(positionIndex) == 0:
        return 0,0
    rows = data[:,positionIndex]
    for row in rows:
        if sum(row) == len(positionIndex):
            PA+=1
    PA/=dataSize
    if PA == 0:
        confidence = 0
    else:
        confidence = PAC/PA
    return PAC,confidence

def checkAntecedentsAndConsequents(LIMIT_ANTECEDENT,LIMIT_CONSEQUENT):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for childIndex in range(len(offspring)):
                child = offspring[childIndex]
                cpChild = numpy.array(child)
                delimitation = int(len(cpChild) / 2)
                position = cpChild[delimitation:] == 1
                rules = cpChild[:delimitation] == 1
                positionIndex = []
                positionConsequent = []
                for ind in range(len(rules)):
                    if rules[ind] == 1 and position[ind] == 1:
                        positionIndex.append(ind)
                    if rules[ind] == 1 and position[ind] == 0:
                        positionConsequent.append(ind)
                positionIndex = numpy.array(positionIndex)
                if len(positionIndex)==0 or len(positionIndex)>LIMIT_ANTECEDENT:
                    cpChild = initIndividual(len(cpChild),LIMIT_ANTECEDENT,LIMIT_CONSEQUENT)
                    for i in range(len(cpChild)):
                        offspring[childIndex][i] = cpChild[i]
                elif len(positionConsequent) == 0 or len(positionConsequent)>LIMIT_CONSEQUENT:
                    cpChild = initIndividual(len(cpChild), LIMIT_ANTECEDENT, LIMIT_CONSEQUENT)
                    for i in range(len(cpChild)):
                        offspring[childIndex][i] = cpChild[i]

            return offspring
        return wrapper
    return decorator


def printRules(hof,data):
    for individual in hof:
        cpChild = numpy.array(individual)
        score = computeMeasures(individual,data)
        delimitation = int(len(cpChild) / 2)
        position = cpChild[delimitation:] == 1
        rules = cpChild[:delimitation] == 1
        positionIndex = []
        positionConsequent = []
        for ind in range(len(rules)):
            if rules[ind] == 1 and position[ind] == 1:
                positionIndex.append(ind)
            if rules[ind] == 1 and position[ind] == 0:
                positionConsequent.append(ind)
        # print('antecedent : ' + str(positionIndex) + ' consequents : '+str(positionConsequent))
        # print(score)

def writeRules(hof,path):
    hofData = []
    scores = []
    for individual in hof:
        cpChild = numpy.array(individual)
        score = computeMeasures(individual, data)
        scores.append(score)
        delimitation = int(len(cpChild) / 2)
        position = cpChild[delimitation:] == 1
        rules = cpChild[:delimitation] == 1
        positionIndex = []
        positionConsequent = []
        for ind in range(len(rules)):
            if rules[ind] == 1 and position[ind] == 1:
                positionIndex.append(ind)
            if rules[ind] == 1 and position[ind] == 0:
                positionConsequent.append(ind)
        hofData.append([positionIndex,positionConsequent,score[0],score[1]])
    df = pandas.DataFrame(hofData,columns=['antecedent','consequent','support','confidence'])
    print(df)
    df.to_csv(path)
    scores = numpy.array(scores)

    return numpy.mean(scores[:,0]),numpy.mean(scores[:,1])


def areEqual(ind1,ind2):
    cp1 = numpy.array(ind1)
    cp2 = numpy.array(ind2)
    delimitation1 = int(len(cp1) / 2)
    position1 = cp1[delimitation1:] == 1
    rules1 = cp1[:delimitation1] == 1
    delimitation2 = int(len(cp2) / 2)
    position2 = cp2[delimitation2:] == 1
    rules2 = cp2[:delimitation2] == 1
    positionIndex1 = []
    positionConsequent1 = []
    positionIndex2 = []
    positionConsequent2 = []
    for ind in range(len(rules1)):
        if rules1[ind] == 1 and position1[ind] == 1:
            positionIndex1.append(ind)
        if rules1[ind] == 1 and position1[ind] == 0:
            positionConsequent1.append(ind)
        if rules2[ind] == 1 and position2[ind] == 1:
            positionIndex2.append(ind)
        if rules2[ind] == 1 and position2[ind] == 0:
            positionConsequent2.append(ind)
    return positionIndex1 == positionIndex2 and positionConsequent1 == positionConsequent2


IND_SIZE=dataSize*2
LIMIT_ANTECEDENT = 2
LIMIT_CONSEQUENT = 1
pbMut = 0.01
difTreshold = 0.01
toolbox = base.Toolbox()






creator.create("Fitness", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox.register("initIndividual", initIndividual, nb=IND_SIZE,LIMIT_ANTECEDENT=LIMIT_ANTECEDENT,LIMIT_CONSEQUENT=LIMIT_CONSEQUENT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.initIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", computeMeasures,data=data)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit)
toolbox.register("select", tools.selNSGA2)
toolbox.decorate("mate", checkAntecedentsAndConsequents(LIMIT_ANTECEDENT, LIMIT_CONSEQUENT))
toolbox.decorate("mutate", checkAntecedentsAndConsequents(LIMIT_ANTECEDENT, LIMIT_CONSEQUENT))

def main(seed=None):
    results = []
    random.seed(seed)

    NGEN = 50
    MU = 100
    CXPB = 0.9
    nbIter = 10
    hof = tools.HallOfFame(300, areEqual)
    stats = tools.Statistics(lambda ind: ind.fitness.values)


    stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    for i in range(nbIter):
        bm = BenchmarkManager('Results/', dataSize)
        hof.clear()
        path = './Results/NSGAII/' + datasetName + str(i)+'.csv'
        result = []
        pop = toolbox.population(n=MU)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
        gen = 0
        previousScores = numpy.array([0,0])
        # Begin the generational process
        scores = numpy.array([1,1])
        isImproving = True
        t1 = time.time()
        while gen < NGEN and isImproving:
            # Vary the population
            hof.update(pop)
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1,pbMut)
                toolbox.mutate(ind2,pbMut)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            scores = record['avg']
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
            # printRules(hof,data)
            difScores = scores - previousScores
            isImproving = difScores.any()>0 and difScores.any()>difTreshold
            if isImproving:
                previousScores = scores
            gen+=1

        t2 = time.time()
        avgsupp,avgconf = writeRules(hof, path)
        result.append(i)
        result.append(t2-t1)
        result.append(avgsupp)
        result.append(avgconf)
        result.append(len(hof))
        firstScore = bm.CompareToExhaustive(data, 'Results/Exhaustive/' + datasetName + str(i) + '.csv', 0.01, 0.01,
                                            'Results/NSGAII/' + datasetName + str(i) + '.csv', i, threshold=0,
                                            nbAntecedent=2, dataset=datasetName)
        result.append(firstScore[-2])
        result.append(firstScore[-1])
        result.append(firstScore[-5])
        results.append(result)
    df = pandas.DataFrame(results,columns=['iter','timeNSGAII','avgSuppNSGAII','avgConfNSGAII','nbRules','NSGAIIFindInFp','FpFindInNSGAII','NSGAIINbNotNull'])
    print(df)
    df.to_csv('./Results/NSGAII/'+datasetName+'_final.csv')


    return pop, logbook



if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, stats = main()

    # pop.sort(key=lambda x: x.fitness.values)

    # print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    # import matplotlib.pyplot as plt
    # import numpy

    # front = numpy.array([ind.fitness.values for ind in pop])
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.axis("tight")
    # plt.show()

