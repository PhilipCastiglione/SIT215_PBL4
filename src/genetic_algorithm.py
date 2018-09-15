import pandas as pd
import numpy as np
import random

class GeneticAlgorithm:
    # initialise with our parameters, and a randomised starting population
    def __init__(self, parameters, features, labels, test_features, test_labels):
        self.population_size = parameters["population_size"]
        self.breeding_ratio = 1 / parameters["breeding_rate"]
        self.crossover_rate = parameters["crossover_rate"]
        self.mutation_rate = parameters["mutation_rate"]
        self.mutation_range = parameters["mutation_range"]
        self.generations = parameters["generations"]
        self.stochastic_parent_selection = parameters["select_parents_stochastically"]

        self.feature_length = features.shape[1]
        self.generation = 0
        self.training_progress = []

        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.population = pd.Series([self._random_chromosome() for i in range(self.population_size)])

    # PUBLIC

    # fit weights of our algorithm, by evolving the population, to the problem
    def fit(self):
        # calculate fitnesses here for initial state
        self.fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome, self.features, self.labels))
        self.test_fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome, self.test_features, self.test_labels))
        for i in range(self.generations):
            self._next_generation()
            self.generation += 1
            self.fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome, self.features, self.labels))
            self.test_fitnesses = self.population.apply(lambda chromosome: self._chromosome_fitness(chromosome, self.test_features, self.test_labels))
            self._update_progress()

    # use to take the best chromosome in the current population (after training) to make
    # predictions over a set of features
    def predict(self, features):
        best_chromosome_idx = self.fitnesses.idxmin()
        best_chromosome = self.population.iloc[best_chromosome_idx].values[0]
        predictions = np.dot(features, best_chromosome)
        return predictions

    # for debugging
    def print_progress(self):
        display = lambda p: print('FITNESS - train_best: {:.2f}, train_mean: {:.2f}, test_best: {:.2f}, test_mean: {:.2f}'.format(p[0], p[1], p[2], p[3]))
        [display(p) for p in self.training_progress]

    # PRIVATE

    # produce a random chromosome, which is a set of weights to map against the feature set
    def _random_chromosome(self):
        _, shape_cols = self.features.shape
        return np.array([random.random() for c in range(shape_cols)])

    # evolve the current generation - the heart of the genetic algorithm
    def _next_generation(self):
        if self.stochastic_parent_selection:
            survivors, parents = self._divide_population_stochastically()
        else:
            survivors, parents = self._divide_population()

        offspring = self._breed(parents)
        self.population = pd.concat([survivors, parents, offspring], ignore_index=True)

    # this approach selects parents stochastically, weighted based on their fitness
    def _divide_population_stochastically(self):
        parent_count = int(len(self.population) / self.breeding_ratio)
        perish_count = parent_count
        survivor_count = len(self.population) - parent_count - perish_count

        sorted_fitnesses = self.fitnesses.sort_values('charges')
        selected_fitnesses = sorted_fitnesses[:-perish_count]
        survivors_fitnesses = selected_fitnesses.sample(survivor_count, weights='charges')
        parents_fitnesses = selected_fitnesses[~selected_fitnesses.index.isin(survivors_fitnesses.index)]

        survivors = self.population[survivors_fitnesses.index]
        parents = self.population[parents_fitnesses.index]
        return survivors, parents

    # simply select the fittest chromosomes to be parents, cull the weakest
    def _divide_population(self):
        parent_count = int(len(self.population) / self.breeding_ratio)
        sorted_fitnesses = self.fitnesses.sort_values('charges')
        parents = self.population[sorted_fitnesses[:parent_count].index]
        survivors = self.population[sorted_fitnesses[parent_count:-parent_count].index]

        return survivors, parents

    # breed parents, producing an equal number of children
    def _breed(self, parents):
        pair_count = int(len(parents) / 2)
        couples = list(zip(parents[:pair_count], parents[pair_count:]))

        # if we have an uneven number of parents, the last one is a hermaphrodite
        if len(parents) % 2 != 0:
            couples.append((parents[-1], parents[-1]))

        children = self._crossover(pd.Series(couples))
        children = self._mutate(children)
        return children

    # performs crossover on a set of couples of parents to generate offspring based on
    # those parents
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

    # randomly mutates particular genes for a set of chromosomes
    def _mutate(self, chromosomes):
        def mutate(chromosome):
            mutation = lambda: 0 if random.random() > self.mutation_rate else random.uniform(-self.mutation_range, self.mutation_range)

            mutations = np.array([mutation() for i in range(self.feature_length)])
            return chromosome - mutations

        return chromosomes.apply(lambda c: mutate(c))

    # calculates the fitness of a chromosome over a set of features and labels using
    # the sum of squared differences
    def _chromosome_fitness(self, chromosome, features, labels):
        predictions = np.dot(features, chromosome)
        differences = pd.DataFrame(labels.charges - predictions)
        squared_differences = differences.apply(lambda diff: diff ** 2)
        return squared_differences.sum()

    # update our progress store, inform the user where we are at
    def _update_progress(self):
        self.training_progress.append((
            self.fitnesses.values.min(),
            self.fitnesses.values.mean(),
            self.test_fitnesses.values.min(),
            self.test_fitnesses.values.mean(),
        ))
        # print('Generation: {}/{}'.format(self.generation, self.generations)) # debug

