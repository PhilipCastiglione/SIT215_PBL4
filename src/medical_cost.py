import pandas as pd

# our data is a set of medical insurance costs for individuals
# each individual is recorded with the following data (features):
#
#   - age (years, discrete value)
#   - sex (binary)
#   - bmi (continuous value)
#   - children (number, discrete value)
#   - smoker (binary)
#   - region (categorical value)
#
# the medical costs of each individual (labels) are recorded
#
#   - charges (continous value)
class MedicalCost:
    DATA_PATH = "./src/data/insurance.csv"

    def __init__(self):
        self.data = pd.read_csv(self.DATA_PATH)

