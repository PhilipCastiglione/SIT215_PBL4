from src.driver import Driver
from src.medical_cost import MedicalCost

def single_run(params):
    print(params)
    medical_cost = MedicalCost()
    driver = Driver(params, medical_cost.data)

    driver.train()

    inputs = medical_cost.data.sample(10)
    driver.predict(inputs)

def grid_search():
    generations = [150, 150, 150] # settled for test, more is better but drops off fast
    population_sizes = [200] # settled for test, more is better but drops off fast
    breeding_rates = [0.4] # so far, .4 is better
    crossover_rates = [0.85, 0.875, 0.9, 0.925, 0.95, 0.975] # so far, higher is better (0.9)
    mutation_rates = [0.2] # so far, higher is better (0.2)
    mutation_ranges = [0.1] # .1 seems better than .2, not super conclusive
    stochastic_selection = [True] # settled

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
    use_grid_search = True

    if use_grid_search:
        grid_search()
    else:
        params = {
            "generations": 10,
            "population_size": 10,
            "breeding_rate": 0.4,
            "crossover_rate": 0.8,
            'mutation_rate': 0.15,
            'mutation_range': 0.2,
            'select_parents_stochastically': True,
        }
        single_run(params)

