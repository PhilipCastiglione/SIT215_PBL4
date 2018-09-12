# Optimisation with Genetic Algorithms

This repository contains a genetic algorithm applied to an optimisation problem for Deakin Artificial and Computational Intelligence PBL Task 4.

We found a modestly sized dataset from [Kaggle](https://www.kaggle.com/datasets) that contains information about individuals and medical insurance costs over a year for those individuals. A genetic algorithm has been built to optimise the weights in a linear algorithm for predicting insurance costs, using the data in the dataset.

DATA ATTRIBUTION: Miri Choi on Kaggle at https://www.kaggle.com/mirichoi0218/insurance

## Solution Details

TODO: technical details

- genetic algorithms (evolution inspiration)
- population initialization
- breeding (stochastically weighted parent selection, vs fittest parent selection)
- crossover (random single point)
- mutation (multiple gene per chromosome)
- fitness cost function (also for predictions)
- hyperparameters and grid search
- modest feature engineering

## Results

The results of the genetic algorithm for optimising the linear algorithm are mixed.

The genetic algorithm successfully converges to a solution using the provided loss minimisation function:

![loss over time](results.png)
_Training progress by generation (loss on a logarithmic scale)_

For both the training and test sets, and for both the minimum loss (fittest chromosome) and average loss (average across population), loss starts out very high and clearly converges.

However, in actual prediction, the generated linear algorithm does not perform terribly well:

```
Predictions:
record 1	 prediction:   -12055	actual:     4466	difference:  -369%
record 2	 prediction:    27101	actual:    47496	difference:   -42%
record 3	 prediction:     1489	actual:    10325	difference:   -85%
record 4	 prediction:    27875	actual:    47291	difference:   -41%
record 5	 prediction:    -2905	actual:     2362	difference:  -222%
record 6	 prediction:   -18134	actual:    24671	difference:  -173%
record 7	 prediction:    13765	actual:    13352	difference:     3%
record 8	 prediction:    24571	actual:    46200	difference:   -46%
record 9	 prediction:    -3914	actual:     7144	difference:  -154%
record 10	 prediction:    29506	actual:    41919	difference:   -29%
```
_Example predictions generated across 10 random examples from the dataset, compared with actual values._

A number of possible explanations exist to explain the discrepancy between effective convergence towards the global minimum and the relatively poor predictions.

First, a linear algorithm may not be a good fit as a model for predicting insurance costs from this data. The relationship between the features in the dataset and the labels may be more complex. Second, the relatively small dataset may provide insufficient information to generate an accurate model. Third, the existing features might not be used in a sufficiently complex way to generate a good model (for example, feature crosses or higher order features are not used). Additional feature engineering might address the problem if this third issue is at play, but the purpose of this assignment was to develop an effective genetic algorithm for optimisation, not to solve the problem of predicting medical insurance costs. We believe successful convergence demonstrates this.

Examples in this section were generated using the following hyperparameters:

```
{'generations': 200, 'population_size': 200, 'breeding_rate': 0.3, 'crossover_rate': 0.9, 'mutation_rate': 0.1, 'mutation_range': 0.1, 'select_parents_stochastically': True}
```

## Usage

This application was developed using Python 3.7.0. This version is recommended.

The following external library dependencies are using in the application, including links and version numbers:

- [pandas](https://pandas.pydata.org/) (0.23.4)
- [scikit-learn](http://scikit-learn.org/) (0.19.2)
- [scipy](https://www.scipy.org/) (1.1.0)
- [numpy](http://www.numpy.org/) (1.15.t1)
- [matplotlib](https://matplotlib.org/) (2.2.3)

Each dependency can be installed using the [pip package manager](https://pypi.org/project/pip/).

To use this solution:

1. Download the code from the [GitHub repository](https://github.com/PhilipCastiglione/SIT215_PBL4)
2. Navigate inside the folder
3. Install the required dependencies (eg `brew install python3` then `pip3 install pandas`, `pip3 install scipy` etc on macOS)
4. Run the code using `python run.py` (this may be `python3 run.py` depending on your setup)

#### Notes

Random seeding of initial states can make a fairly significant difference to training results, but this differences erodes over larger generation and population sizes to become relatively negligible.

Debug options are available in the code for extended logging. They're implemented pretty naively, just uncomment the lines ending in a comment `# debug`.
