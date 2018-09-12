# Optimisation with Genetic Algorithms

This repository contains a genetic algorithm applies in an optimisation problem for Deakin Artificial and Computational Intelligence PBL Task 3.

We found a modestly sized dataset from [Kaggle](https://www.kaggle.com/datasets) that contains information about individuals and medical insurance costs over a year for those individuals. A genetic algorithm has been built to optimise the weights in a linear algorithm for predicting insurance costs, using the data in the dataset.

DATA ATTRIBUTION: Miri Choi on Kaggle at https://www.kaggle.com/mirichoi0218/insurance

## Solution Details

TODO: technical details

## Results

TODO: predictions aren't actually very good

## Usage

This application was developed using Python 3.7.0. This version is recommended.

The following external library dependencies are using in the application, including links and detailed version numbers:

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

