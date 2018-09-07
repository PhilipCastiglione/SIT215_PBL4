import pandas as pd
import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, parameters, features, labels, test_features, test_labels):
        self.population_size = parameters["population_size"]
        self.breeding_ratio = 1 / parameters["breeding_rate"]
        self.crossover_rate = parameters["crossover_rate"]
        self.mutation_rate = parameters["mutation_rate"]
        self.mutation_range = parameters["mutation_range"]
        self.generations = parameters["generations"]

        self.feature_length = features.shape[1]
        self.generation = 0
        self.training_progress = []

        self.features = features
        self.labels = labels
        self.population = pd.Series([self._random_chromosome() for i in range(self.population_size)])

    # PUBLIC

    def fit(self):
        # calculate fitnesses here for initial state
        self.fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome))
        for i in range(self.generations):
            self._next_generation()
            self.generation += 1
            self.fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome))
            self._update_progress()

    def predict(self, features):
        best_chromosome_idx = self.fitnesses.idxmax()
        best_chromosome = self.population.iloc[best_chromosome_idx].values[0]
        predictions = np.dot(features, best_chromosome)
        return predictions

    # for debugging
    def print_progress(self):
        display = lambda p: print('FITNESS - best: {:.5f}, mean: {:.5f}'.format(p[0], p[1]))
        [display(p) for p in self.training_progress]

    # PRIVATE

    def _random_chromosome(self):
        _, shape_cols = self.features.shape
        return np.array([random.random() for c in range(shape_cols)])

    def _next_generation(self):
        weakest, middle, parents = self._divide_population()
        offspring = self._breed(parents)
        self.population = pd.concat([middle, parents, offspring], ignore_index=True)

    # TODO: (optional) select parents stochastically, based on fitness, rather than just taking the best ones
    def _divide_population(self):
        parent_count = int(len(self.population) / self.breeding_ratio)

        sorted_fitnesses = self.fitnesses.sort_values('charges')
        parents = self.population[sorted_fitnesses[:parent_count].index]
        middle = self.population[sorted_fitnesses[parent_count:-parent_count].index]
        weakest = self.population[sorted_fitnesses[-parent_count:].index]

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

        children = self._crossover(pd.Series(couples))
        children = self._mutate(children)
        return children

    def _crossover(self, couples):
        def crossed_children(couple):
            x_point = 0 if random.random() > self.crossover_rate else random.randrange(0, self.feature_length + 1)
            child1 = np.concatenate((couple[0][:x_point], couple[1][x_point:]), axis=None)
            child2 = np.concatenate((couple[1][:x_point], couple[0][x_point:]), axis=None)
            return (child1, child2)

        couple_children = couples.apply(lambda c: crossed_children(c))

        children = []
        for cc in couple_children:
            children.append(cc[0])
            children.append(cc[1])

        return pd.Series(children)

    def _mutate(self, children):
        def mutate(child):
            mutation = lambda: 0 if random.random() > self.mutation_rate else random.uniform(-self.mutation_range, self.mutation_range)

            mutations = np.array([mutation() for i in range(self.feature_length)])
            return child - mutations

        return children.apply(lambda c: mutate(c))

    # update our progress store, inform the user where we are at
    def _update_progress(self):
        self.training_progress.append((self.fitnesses.values.min(), self.fitnesses.values.mean()))
        # TODO: assess quality against test
        print('Generation: {}/{}'.format(self.generation, self.generations))

