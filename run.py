from src.driver import Driver
from src.medical_cost import MedicalCost

# entry point; you can change the hyperparameters here
if __name__ == '__main__':
    params = {
        "generations": 10,
        "population_size": 10,
        "breeding_rate": 0.2,
        "crossover_rate": 0.7,
        'mutation_rate': 0.08,
        'mutation_range': 0.2,
    }
    print(params)

    medical_cost = MedicalCost()
    driver = Driver(params, medical_cost.data)
    driver.train()

