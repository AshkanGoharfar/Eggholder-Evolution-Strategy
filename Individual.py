import random as rnd
import math
import random


class Individual:
    def __init__(self, an_individual):
        self.an_individual = an_individual

    best_gen = [1, 1]
    fitness = 0
    individual_fitness = 0

