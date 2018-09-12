import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from src.genetic_algorithm import GeneticAlgorithm

class Driver:
    # initialise with hyperparameters, data
    def __init__(self, params, data):
        self.params = params
        self.raw_data = data

    # PUBLIC

    # this executes a training run, transforming our data and passing it to the genetic algorithm to fit
    # we cache our scalers and our mappings so we can use them for transforming data, for booth
    # training and predictions
    def train(self):
        raw_features = pd.DataFrame(self.raw_data, columns=self.raw_data.columns.drop('charges'))
        self.feature_scalers = [(preprocessing.MinMaxScaler() ,'age'), (preprocessing.MinMaxScaler() ,'bmi')]
        self.mappings = self._extract_category_mappings(raw_features, ['sex', 'smoker', 'region'])
        features = self._transform_features(raw_features, self.mappings, self.feature_scalers)

        raw_labels = pd.DataFrame(self.raw_data.charges)
        self.label_scaler = (preprocessing.MinMaxScaler(), 'charges')
        labels = pd.DataFrame(index=raw_labels.index)
        labels['charges'] = self._normalize(raw_labels, [self.label_scaler[1]], self.label_scaler[0])
        X_train, X_test, y_train, y_test = train_test_split(features, labels)

        self.genetic_algorithm = GeneticAlgorithm(self.params, X_train, y_train, X_test, y_test)
        self.genetic_algorithm.fit()

    # generate charts to display training progress over the train and test sets
    def report(self):
        # self.genetic_algorithm.print_progress() # debug
        columns = ['train_min', 'train_mean', 'test_min', 'test_mean']
        progress = pd.DataFrame(self.genetic_algorithm.training_progress, columns=columns)
        progress.plot(logy=True, legend=True, title="Training progress: loss by generation")
        plt.show()

    # generate predictions for a set of input values, transforming input to match
    # the feature set representation of a chromosome, and converting the prediction
    # back to the original scale
    def predict(self, inputs):
        raw_features = pd.DataFrame(inputs, columns=inputs.columns.drop('charges'))
        features = self._transform_features(raw_features, self.mappings, self.feature_scalers, fit=False)

        normalized_predictions = self.genetic_algorithm.predict(features)
        denormalized_predictions = [self._denormalize(p, self.label_scaler[0]) for p in normalized_predictions]
        predictions = np.array(denormalized_predictions)

        labels = np.array(inputs.charges)

        diffs = (predictions - labels) / labels * 100

        print("Predictions:")
        for i in range(inputs.shape[0]):
            prediction = str(int(predictions[i]))
            label = str(int(labels[i]))
            diff = str(int(diffs[i]))
            print("record {}\t prediction: {}\tactual: {}\tdifference: {}%".format(i + 1, prediction.rjust(8), label.rjust(8), diff.rjust(5)))

    # PRIVATE

    # create mappings for a set of values in a categorical feature column to binary
    # representations so we can train on them effectively as genes
    def _extract_category_mappings(self, dataframe, category_columns):
        to_map = lambda uniques: {v:k[0] for k, v in np.ndenumerate(uniques)}
        return [(to_map(dataframe[col].unique()), col) for col in category_columns]

    # normalize continuous data to make training more efficient ands weights similar
    def _normalize(self, series, columns, scaler, fit=True):
        return scaler.fit_transform(series[columns]) if fit else scaler.transform(series[columns])

    # denormalize using a fitted scaler to convert values back to the domain
    def _denormalize(self, value, scaler):
        return scaler.inverse_transform(value)[0][0]

    # transform our features, mapping categorical columns and scaling continuous values
    # NB: children is a low valued discrete feature so we leave it alone
    def _transform_features(self, dataframe, mappings, scalers, fit=True):
        transformed_df = pd.DataFrame(dataframe, columns=['children'])

        for mapping, name in mappings:
            for i in mapping.values():
                transformed_df[name + str(i)] = dataframe[name].apply(lambda x: int(mapping[x] == i))

        for scaler, name in scalers:
            transformed_df[name] = self._normalize(dataframe, [name], scaler, fit=fit)

        return transformed_df

