from src.driver import Driver
from src.medical_cost import MedicalCost

# entry point; you can change the hyperparameters here
if __name__ == '__main__':
    params = {
        "generations": 100,
        "population_size": 100,
        "breeding_rate": 0.2,
        "crossover_rate": 0.6,
        'mutation_rate': 0.08,
        'mutation_range': 0.2,
    }

    medical_cost = MedicalCost()
    driver = Driver(params, medical_cost.data)
    driver.train()

