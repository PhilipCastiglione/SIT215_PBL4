from src.driver import Driver
from src.medical_cost import MedicalCost

# entry point; you can change the hyperparameters here
if __name__ == '__main__':
    params = {
        "generations": 1000,
        "population_size": 200,
        "breeding_rate": 0.3,
        "crossover_rate": 0.8,
        'mutation_rate': 0.1,
        'mutation_range': 0.2,
    }
    print(params)

    medical_cost = MedicalCost()
    driver = Driver(params, medical_cost.data)

    driver.train()

    inputs = medical_cost.data.sample(10)
    driver.predict(inputs)
