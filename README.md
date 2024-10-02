# DynamoRep features for problem and algorithm classification repository

This repository contains the code for analyzing the performance of the features for the problem classification task and the algorithm classification task in the following settings:
    - with different feature calculation budgets
    - with and without scaling the objective function values (performed before feature calculation)
    - with and without differencing the features (performed after feature calculation)
    - using a single descriptive statistic or using all descriptive statistics for calculating the DynamoRep features
    - using only the decision variables (x-values) or only the objective function values (y-values) or both

### Setup
The following repository requires the R language for ELA feature calculation, and the python language for the rest of the logic.
In our experiments, R version 4.2.2 and python version 3.9.7 were used.

For the ELA feature calculation, the "flacco" package should be installed in R. This can be done using the command:
`R -e 'install.packages("flacco", dependencies = TRUE, repos = "http://cran.us.r-project.org", Ncpus = -1)'`

The libraries used in the python scripts are listed in the requirements.txt file and can be installed with the command:
`pip install -r requirements.txt`

### Data collection
- The scripts algorithms.py can be used to run the DE, ES and PSO algorithms on the BBOB problems to collect the trajectory data needed for the classification tasks. 
They receive as input two parameters:
  - $1 - seed - the random seed which should be fixed for the algorithm execution
  - $2 - dimension - the dimension of the problems on which to run the algorithms
The resulting algorithm trajectory is saved in a file named  f'algorithm_run_data/{algorithm_name}_dim_{dimension}_seed_{seed}.csv'

-The script scale_y.py is used to scale the objective function values for each trajectory in the f'algorithm_run_data/{algorithm_name}_dim_{dimension}_seed_{seed}.csv' files, and saves the scaled trajectories in the folder f'algorithm_run_data_normalized/{algorithm_name}_dim_{dimension}_seed_{seed}.csv'.

- The scripts ela_iteration.R, and ela_iteration_normalized.R are used to calculate the ELA features for the BBOB problems. The scripts calculate the features for a single run of a single algorithm on a single BBOB problem class. This is done in order to enable calculating the features for the different BBOB problem classes in parallel. 

They receive as input the following parameters:
  - $1 - algorithm - the name of the algorithm for which to calculate the features
  - $2 - seed - the seed of the algorithm execution
  - $3 - dimension - the problem dimension
  - $4 - problem_id - the id of the BBOB problem (from 1 to 24) for which to calculate the feature.
The notebook merge_iteration_ela.ipynb can then be used to merge the files produced for the individual BBOB files into a single one.
This concludes the data collection process.

## Scripts using the DynamoRep features

### Problem classification
- problem_classification.py - This script performs the problem classification task with the DynamoRep features with different feature calculation budgets using all features
- problem_classification_x_y.py - This script performs the problem classification task with the DynamoRep features with different feature calculation budgets, and using only the decision variables (x-values) or only the objective function values (y-values)
- problem_classification_single_statistic.py - This script performs the problem classification task with the DynamoRep features with different feature calculation budgets, using a single descriptive statistic

- problem_classification_ela.py -  This script performs the problem classification task with the iteration ELA features with different feature calculation budgets

### Algorithm classification
- algorithm_classification.py - This script performs the algorithm classification task with the DynamoRep features with different feature calculation budgets using all features
- algorithm_classification_x_y.py - This script performs the algorithm classification task with the DynamoRep features with different feature calculation budgets, and using only the decision variables (x-values) or only the objective function values (y-values)
- algorithm_single_statistic.py - This script performs the algorithm classification task with the DynamoRep features with different feature calculation budgets, using a single descriptive statistic

- algorithm_classification_ela.py -  This script performs the algorithm classification task with the iteration ELA features with different feature calculation budgets
### Script Parameters
All 8 scripts receive as input the following parameter (all parameters need to be specified in the following order):
- $1 - algorithm_name(s) - a single algorithm name (for problem classification) or multiple algorithm names separated by "-", example: "DE" (for problem classification) or "DE-ES-PSO" (for algorithm classification)
- $2 - train_on_seed, whether to train on one seed, always "true" in our experiments
- $3 - difference, whether to difference the features, example: "true"/"false"
- $4 - the starting and ending iteration to use for calculating the features, example "0-20" (means the feature calculation budget is 20 iterations)
- $5 - the problem dimension, example: "5"
- $6 - normalize_y - whether to use the features calculated on scaled objective function values or not, example: "true"/"false"


### Algorithm selection 
- algorithm_selection.py - This script performs the algorithm selection task with the DynamoRep features with different feature calculation budgets using all features
- algorithm_selection_iteration_ela.py - This script performs the algorithm selection task with the iteration ELA features with different feature calculation budgets using all features
- algorithm_selection_ela_static.py - This script performs the algorithm selection task with the static ELA features with different feature calculation budgets using all features


### Shell scripts
The shell scripts run_problem_classification_ela.sh, run_problem_classification_single.sh, run_problem_classification.sh, run_algorithm_classification_ela.sh, run_algorithm_classification_single.sh, run_algorithm_classification.sh, run_as.sh, run_as_static_ela.sh can be used to run the python scripts with the parameters used in the paper.



### Visualizations and Analysis of results
The remaining notebooks are used to produce the figures for the paper.