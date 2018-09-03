import pandas as pd
from sklearn.model_selection import train_test_split

class MedicalCost:
    DATA_PATH = "./src/data/insurance.csv"

    def __init__(self):
        all_data = pd.read_csv(self.DATA_PATH)
        self.train, self.test = train_test_split(all_data)

