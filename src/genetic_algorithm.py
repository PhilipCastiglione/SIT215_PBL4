import pandas as pd
import numpy as np
import random

# TODO:
# - use less raw python, more pandas/numpy
# - assess quality against test
class GeneticAlgorithm:
    def __init__(self, parameters, features, labels, test_features, test_labels):
        self.population_size = parameters["population_size"]
        self.breeding_ratio = 1 / parameters["breeding_rate"]
        self.crossover_rate = parameters["crossover_rate"]
        self.mutation_rate = parameters["mutation_rate"]
        self.mutation_range = parameters["mutation_range"]
        self.generations = parameters["generations"]

        self.generation = 0
        self.training_progress = []

        self.features = features
        self.labels = labels
        self.population = pd.Series([self._random_chromosome() for i in range(self.population_size)])

    # PUBLIC

    def fit(self):
        # cache fitnesses here for initial state
        self.fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome))
        for i in range(self.generations):
            self._next_generation()
            self.generation += 1
            self.fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome))
            self._update_progress()

    # for debugging
    def print_progress(self):
        display = lambda p: print('FITNESS - best: {:.5f}, mean: {:.5f}'.format(p[0], p[1]))
        [display(p) for p in self.training_progress]

    # PRIVATE

    def _random_chromosome(self):
        _, shape_cols = self.features.shape
        return [random.random() for c in range(shape_cols)]

    def _next_generation(self):
        weakest, middle, parents = self._divide_population()
        offspring = self._breed(parents)
        self.population = pd.concat([middle, parents, offspring], ignore_index=True)

    def _divide_population(self):
        parent_count = int(len(self.population) / self.breeding_ratio)

        # TODO: select parents stochastically, based on fitness, rather than just taking the best ones
        parents = self.population[self.fitnesses.sort_values('charges')[:parent_count].index]
        middle = self.population[self.fitnesses.sort_values('charges')[parent_count:-parent_count].index]
        weakest = self.population[self.fitnesses.sort_values('charges')[-parent_count:].index]

        return weakest, middle, parents

    def _chromosome_fitness(self, chromosome):
        predictions = np.dot(self.features, chromosome)
        differences = pd.DataFrame(self.labels.charges - predictions)
        squared_differences = differences.apply(lambda diff: diff ** 2)
        return squared_differences.sum()

    def _breed(self, parents):
        pair_count = int(len(parents) / 2)
        couples = list(zip(parents[:pair_count], parents[pair_count:]))

        # if we have an uneven number of parents, the last one is a hermaphrodite
        if len(parents) % 2 != 0:
            couples.append((parents[-1], parents[-1]))

        children = self._crossover(couples)
        self._mutate(children)
        return pd.Series(children)

    def _crossover(self, couples):
        children = []
        for couple in couples:
            p1, p2 = couple
            if random.random() < self.crossover_rate:
                x_point = random.randrange(0, len(p1) + 1)
                children.append(p1[:x_point] + p2[x_point:])
                children.append(p2[:x_point] + p1[x_point:])
            else:
                children.append(p1)
                children.append(p2)
        return children

    def _mutate(self, children):
        for child in children:
            for feature_idx, feature in enumerate(child):
                if random.random() < self.mutation_rate:
                    m = random.uniform(-self.mutation_range, self.mutation_range)
                    child[feature_idx] += m
        return children

    # update our progress store, inform the user where we are at
    def _update_progress(self):
        self.training_progress.append((self.fitnesses.values.min(), self.fitnesses.values.mean()))
        print('Generation: {}/{}'.format(self.generation, self.generations))

