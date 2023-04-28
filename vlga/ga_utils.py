import numpy as np

from vlga.chromosome import Chromosome


def crossover(chromosome1, chromosome2):
    config1 = chromosome1.get_config()
    config2 = chromosome2.get_config()

    size1 = config1.shape[0]
    size2 = config2.shape[0]

    min_size = min(size1, size2)

    random_position = np.random.randint(min_size - 1)

    # 1A: [0, 1, ..., random_position]; 1B: [random_position + 1, size1 - 1]
    # 2A: [0, 1, ..., random_position]; 2B: [random_position + 1, size2 - 1]

    part_1a = config1[:random_position + 1]
    part_1b = config1[random_position + 1:]

    part_2a = config2[:random_position + 1]
    part_2b = config2[random_position + 1:]

    # concat operator return a new array instead of
    # referencing the original array
    new_config1 = np.concatenate((part_1a, part_2b))
    new_config2 = np.concatenate((part_2a, part_1b))

    new_chromosome1 = Chromosome(new_config1, None)
    new_chromosome2 = Chromosome(new_config2, None)

    return new_chromosome1, new_chromosome2


def mutate(chromosome):
    config = chromosome.get_config()
    size = config.shape[0]

    random_position = np.random.randint(size)
    config[random_position] = 1 - config[random_position]

    chromosome.set_config(config)
    chromosome.set_fitness(None)
