import datetime

import numpy as np
import pdb

from vlga.chromosome import Chromosome
from enum import Enum

from vlga.ga_utils import crossover, mutate
from vlga.result_collector import ResultCollector


class SelectionType(str, Enum):
    ROULETTE = "roulette"
    RANDOM = "random"


class VLGA:

    def __init__(self, chromosome_chunk_size=1, max_n_chunk=10,  # variable-length params
                 n_generations=200, prob_crossover=0.9, prob_mutation=0.1, n_population=100,  # normal GA params
                 selection_type=SelectionType.ROULETTE, random_state=1, log_detail=True):
        """
        chromosome:
            length = n_chunk * chromosome_chunk_size
            |xxxx xxxx xxxx| -> length = 12, n_chunk = 3, chunk_size = 4
        """
        self.log_detail = log_detail
        self.selection_type = selection_type
        self.max_n_chunk = max_n_chunk
        self.chromosome_chunk_size = chromosome_chunk_size
        self.n_population = n_population
        self.prob_mutation = prob_mutation
        self.prob_crossover = prob_crossover
        self.n_generations = n_generations

        np.random.seed(random_state)
        self.population = []
        self.result_collector = ResultCollector(log_detail)

    def dummy_solve(self, problem):
        n_chunk = np.random.randint(1, self.max_n_chunk)
        chromosome_size = n_chunk * self.chromosome_chunk_size

        random_candidate = np.random.rand(chromosome_size)
        return random_candidate

    def solve(self, problem):

        self.initialize_population(problem)

        for i_generation in range(self.n_generations):
            print(datetime.datetime.now(), ' At generation {}'.format(i_generation))

            population_fitness = np.array([c.get_fitness() for c in self.population])

            population_mean = population_fitness.mean()
            population_best = population_fitness[0]
            self.result_collector.add_generation_average_fitness(population_mean)
            self.result_collector.add_generation_best_fitness(population_best)

            print('Current generation best: {}'.format(population_best))
            print('Current generation mean: {}'.format(population_mean))

            selection_probability = population_fitness / population_fitness.sum()

            while True:
                # 1. crossover process
                if self.selection_type == SelectionType.RANDOM:
                    id1, id2 = np.random.choice(self.n_population, 2, replace=False)
                else:
                    id1, id2 = np.random.choice(self.n_population, 2, replace=False, p=selection_probability)

                r_crossover = np.random.rand()
                if r_crossover < self.prob_crossover:
                    new_chromosome1, new_chromosome2 = crossover(self.population[id1], self.population[id2])

                    self.population.append(new_chromosome1)
                    self.population.append(new_chromosome2)

                    # 2. mutation process
                    # NOTE: new_chromosome1 & new_chromosome2 still refer to those in population
                    r_mutation = np.random.rand()
                    if r_mutation < self.prob_mutation:
                        # Apply mutation on the 2 new offsprings
                        mutate(new_chromosome1)
                        mutate(new_chromosome2)

                    # 3. recalculate fitness for the 2 new offsprings
                    config1 = new_chromosome1.get_config()
                    config2 = new_chromosome2.get_config()

                    fitness1 = problem.fitness(config1)
                    fitness2 = problem.fitness(config2)

                    new_chromosome1.set_fitness(fitness1)
                    new_chromosome2.set_fitness(fitness2)

                # print(len(self.population))
                if len(self.population) >= 2 * self.n_population:
                    break

            # 4. natural selection 2*L -> L
            self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
            del self.population[self.n_population:]

        return self.population[0].get_config(), self.population[0].get_fitness()

    def initialize_population(self, problem):
        print('Initialize population')
        self.population = []
        for i_population in range(self.n_population):
            n_chunk = np.random.randint(1, self.max_n_chunk + 1)  # interval [1, max_n_chunk]
            chromosome_size = n_chunk * self.chromosome_chunk_size

            config = np.random.rand(chromosome_size)
            fitness = problem.fitness(config)

            chromosome = Chromosome(config, fitness)

            self.population.append(chromosome)

        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)

        print('Done initializing')

    def get_result_collector(self):
        return self.result_collector
