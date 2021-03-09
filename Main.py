from Operations import *
import matplotlib.pyplot as plt
import numpy as np

population = []

populationSize = 10
number_of_generation = 0
tornumentSize = 10
individualSize = 10
mutation_rate = 0.2
# for raising diversity you increase normal_distribution_value_1
normal_distribution_value_1 = 20
normal_distribution_value_2 = 80
generation = []
best_individual = []
best_answer_fitness = 0
best_answer_gen = [1, 2]
counter = 2000

if __name__ == '__main__':
    # Generate new populations
    for i in range(populationSize):
        initialed_individual = []
        for j in range(individualSize):
            initialed_individual.append([np.random.uniform(-512, 512), np.random.uniform(-512, 512)])
        generation.append(Individual(initialed_individual))

    report = []
    report_1 = []

    flag_terminate = 0
    while flag_terminate == 0 and number_of_generation < counter:
        for individual in generation:
            best_gens = individual_best_gen(individual)
            individual.individual_fitness = individual_mean_fitness(individual)
            individual.best_gen = best_gens
            individual.fitness = fitness(best_gens)
        generation_sorted = sorted(generation, key=operator.attrgetter('fitness'), reverse=True)
        population_selected = generation_sorted[0:tornumentSize]
        number_of_generation += 1
        new_generation = generate_crossover(population_selected, normal_distribution_value_1,
                                            normal_distribution_value_2,
                                            individualSize, mutation_rate)
        # print(new_generation[6].an_individual)
        all_gens = new_generation + generation
        all_gens_sorted = sorted(all_gens, key=operator.attrgetter('individual_fitness'), reverse=True)
        generation = all_gens_sorted[0:populationSize]
        for i in range(len(generation)):
            if generation[i].fitness > best_answer_fitness:
                best_answer_fitness = generation[i].fitness
                best_answer_gen = generation[i].best_gen
        if best_answer_fitness > 1000:
            # print('The best best answer : ')
            # print(best_answer_fitness)
            # print(best_answer_gen)
            flag_terminate = 1
        # print(np.mean(np.array([item.individual_fitness for item in generation])))
        report.append({'generation': number_of_generation,
                       'fitness': np.mean(np.array([item.individual_fitness for item in generation]))})
        print(report)
        report_1.append({'generation': number_of_generation,
                         'fitness': best_answer_fitness})

    print('best_answer fitness : ')
    print(best_answer_fitness)
    print('best_answer generation : ')
    print(best_answer_gen)

    # x axis values
    x = []
    # corresponding y axis values
    y = []
    print('All generations :')
    print(report)
    for item in report:
        x.append(item['generation'])
        y.append(item['fitness'])

    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Number of the generations')
    plt.ylabel('Best answer fitness in a generation')
    plt.title('Best answers fitness in every generation')
    plt.plot()
    plt.show()

    # x axis values
    x = []
    # corresponding y axis values
    y = []
    for item in report_1:
        x.append(item['generation'])
        y.append(item['fitness'])

    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Number of the generations')
    plt.ylabel('Mean fitness')
    plt.title('Mean fitness in every generation')
    # plt.title('Comaparison dataset columns')
    plt.plot()
    plt.show()

    plt.plot()
    plt.show()
    plt.plot(x, y)
