import random
import math
from Individual import Individual
import numpy as np
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
import operator
import matplotlib.pyplot as plt
import itertools


def individual_mean_fitness(individual):
    each_individual_fitness = 0
    np.mean(individual.an_individual)
    for gen in individual.an_individual:
        each_individual_fitness += fitness(gen)
    return each_individual_fitness / len(individual.an_individual)


def fitness(gen):
    return 1 / np.absolute(
        -959.6407 - (-1 * (gen[1] + 47) * np.sin(np.sqrt(np.absolute(gen[1] + gen[0] / 2 + 47))) - gen[0] * np.sin(
            np.sqrt(np.absolute(gen[0] - (gen[1] + 47))))))


def individual_best_gen(individual):
    best_fitness = -100000
    best_gens = []
    for gen in individual.an_individual:
        fit = fitness(gen)
        if fit > best_fitness:
            best_gens = gen
            best_fitness = fit
    return best_gens


def remove_iterated_gens(new_child):
    gens = []
    for gen in new_child:
        if gen not in gens:
            gens.append(gen)
        else:
            del new_child[new_child.index(gen)]
    return new_child


def remove_iterated_items(list):
    items = []
    for item in list:
        if item not in items:
            items.append(item)
        else:
            del list[list.index(item)]
    return list


def mutation(new_child, normal_distribution_value_1, normal_distribution_value_2, individualSize, mutation_rate):
    normal_distribution_list_1 = np.random.normal(loc=0, scale=normal_distribution_value_1, size=(individualSize, 2))
    normal_distribution_list_2 = np.random.normal(loc=0, scale=normal_distribution_value_2, size=(individualSize, 2))
    normal_distribution_list_3 = np.random.normal(loc=0, scale=10,
                                                  size=(individualSize, 2))
    for i in range(len(new_child)):
        # mutation rate
        chance = np.random.uniform(0, 1)
        flag_state_1 = 0
        flag_state_2 = 0
        flag_state_3 = 0
        flag_state_4 = 0
        if chance < mutation_rate:
            better_value = [np.random.uniform(-512, 512), np.random.uniform(-512, 512)]
            better_value[0] = new_child[i][0] + normal_distribution_list_3[i][0]
            better_value[1] = new_child[i][1] + normal_distribution_list_3[i][1]
            if np.absolute(better_value[0]) < 513 and np.absolute(
                    better_value[1]) < 513:
                new_child[i] = better_value
                flag_state_6 = 1
        else:
            better_value = [0, 0]
            better_value[0] = new_child[i][0] + normal_distribution_list_1[i][0]
            better_value[1] = new_child[i][1] + normal_distribution_list_1[i][1]
            chance_1 = 1
            # chance_1 = np.random.uniform(0, 1)
            if fitness(better_value) > fitness(new_child[i]) and np.absolute(better_value[0]) < 513 and np.absolute(
                    better_value[1]) < 513 and chance_1 > mutation_rate:
                new_child[i] = better_value
                flag_state_1 = 1
            chance_2 = 1
            # chance_2 = np.random.uniform(0, 1)
            if flag_state_1 == 0 and chance_2 > mutation_rate:
                better_value = [0, 0]
                better_value[0] = new_child[i][0] - normal_distribution_list_1[i][0]
                better_value[1] = new_child[i][1] + normal_distribution_list_1[i][1]
                if fitness(better_value) > fitness(new_child[i]) and np.absolute(better_value[0]) < 513 and np.absolute(
                        better_value[1]) < 513:
                    new_child[i] = better_value
                    flag_state_2 = 1
            chance_3 = 1
            # chance_3 = np.random.uniform(0, 1)
            if flag_state_1 == 0 and flag_state_2 == 0 and chance_3 > mutation_rate:
                better_value = [0, 0]
                better_value[0] = new_child[i][0] + normal_distribution_list_1[i][0]
                better_value[1] = new_child[i][1] - normal_distribution_list_1[i][1]
                if fitness(better_value) > fitness(new_child[i]) and np.absolute(better_value[0]) < 513 and np.absolute(
                        better_value[1]) < 513:
                    new_child[i] = better_value
                    flag_state_3 = 1
            chance_4 = 1
            # chance_4 = np.random.uniform(0, 1)
            if flag_state_1 == 0 and flag_state_2 == 0 and flag_state_3 == 0 and chance_4 > mutation_rate:
                better_value = [0, 0]
                better_value[0] = new_child[i][0] - normal_distribution_list_1[i][0]
                better_value[1] = new_child[i][1] - normal_distribution_list_1[i][1]
                if fitness(better_value) > fitness(new_child[i]) and np.absolute(better_value[0]) < 513 and np.absolute(
                        better_value[1]) < 513:
                    new_child[i] = better_value
                    flag_state_4 = 1
            chance_5 = 1
            # chance_5 = np.random.uniform(0, 1)
            if flag_state_1 == 0 and flag_state_2 == 0 and flag_state_3 == 0 and flag_state_4 == 0 and chance_5 > mutation_rate:
                better_value = [0, 0]
                better_value[0] = new_child[i][0] + normal_distribution_list_2[i][0]
                better_value[1] = new_child[i][1] + normal_distribution_list_2[i][1]
                if np.absolute(better_value[0]) < 513 and np.absolute(
                        better_value[1]) < 513:
                    new_child[i] = better_value
                    flag_state_5 = 1
            if np.absolute(new_child[i][0]) > 513 or np.absolute(
                    new_child[i][1]) > 513:
                print('holly !!!!!!!!!!!!!!!')
                print(new_child[i])
    return new_child


def generate_crossover(individuals, normal_distribution_value_1, normal_distribution_value_2, individualSize,
                       mutation_rate):
    new_pop = []
    for i in range(len(individuals)):
        first_individual = individuals[(i + 1) % len(individuals)]
        second_individual = individuals[i]
        parents_gens = []
        for j in range(2 * individualSize):
            if j > individualSize - 1:
                parents_gens.append([second_individual.an_individual[j % individualSize],
                                     fitness(second_individual.an_individual[j % individualSize])])
            else:
                parents_gens.append([first_individual.an_individual[j], fitness(first_individual.an_individual[j])])
        parents_gens = sorted(parents_gens, key=itemgetter(1), reverse=True)
        for i in range(len(parents_gens)):
            parents_gens[i] = parents_gens[i][0]
        child_individual = parents_gens[:individualSize]
        new_indiv = mutation(child_individual, normal_distribution_value_1, normal_distribution_value_2,
                             individualSize, mutation_rate)
        new_individual = Individual(new_indiv)
        new_individual_best_gen = individual_best_gen(new_individual)
        new_individual.fitness = fitness(new_individual_best_gen)
        new_individual.best_gen = new_individual_best_gen
        new_individual.individual_fitness = individual_mean_fitness(new_individual)
        new_pop.append(new_individual)
    return new_pop
