from src.driver import Driver
from src.medical_cost import MedicalCost

# takes parameters and performs a run of training and predictions
def single_run(params):
    print(params)
    medical_cost = MedicalCost()
    driver = Driver(params, medical_cost.data)

    driver.train()
    driver.report()

    inputs = medical_cost.data.sample(10)
    driver.predict(inputs)

# used for a grid search over hyperparameters, performing training runs
# over the provided combinations - for optimisation of hyperparameters
def grid_search():
    generations = [50, 100, 250]
    population_sizes = [50, 100, 200]
    breeding_rates = [0.2, 0.3, 0.4]
    crossover_rates = [0.7, 0.8, 0.9]
    mutation_rates = [0.1, 0.2, 0.3]
    mutation_ranges = [0.1, 0.2, 0.3]
    stochastic_selection = [True, False]

    for generation in generations:
        for population_size in population_sizes:
            for breeding_rate in breeding_rates:
                for crossover_rate in crossover_rates:
                    for mutation_rate in mutation_rates:
                        for mutation_range in mutation_ranges:
                            for selection in stochastic_selection:
                                params = {
                                    "generations": generation,
                                    "population_size": population_size,
                                    "breeding_rate": breeding_rate,
                                    "crossover_rate": crossover_rate,
                                    'mutation_rate': mutation_rate,
                                    'mutation_range': mutation_range,
                                    'select_parents_stochastically': selection,
                                }
                                single_run(params)

# entry point; you can change the hyperparameters here
# and choose whether to initiate a single run or a grid
# search over the hyperparameters, for optimisation
if __name__ == '__main__':
    # this toggle is available for the hyperparameter grid search
    use_grid_search = False

    if use_grid_search:
        grid_search()
    else:
        params = {
            "generations": 200,
            "population_size": 200,
            "breeding_rate": 0.3,
            "crossover_rate": 0.9,
            'mutation_rate': 0.1,
            'mutation_range': 0.1,
            'select_parents_stochastically': True,
        }
        single_run(params)

