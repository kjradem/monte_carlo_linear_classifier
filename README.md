# Monte Carlo for a linear classifier
Monte Carlo simulations to optimize a linear classifier.

## Background
### Monte Carlo Simulations
Monte Carlo simulations are a method to randomly test all options for a given situation, in this case all possible configurations of coefficients for the formula __y = a * Polarity + b * Hydrophobicity + c__ with an upper and lower limit for coefficients. More on ![Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method).
### Data
The data consists of a classification of proteins as either membrane proteins (0) or non-membrane proteins (1) with the attributes polarity and hydrophobicity.
### Linear classification
The goal is to generate a formula for linear classification that can accurately separate the two protein classes, as shown below:

![Linear classification](https://user-images.githubusercontent.com/24732704/41862683-e8616346-78a4-11e8-94e2-d5aa90661943.png)

There are many formulas that will accomplish this, however running all possible configurations with limits should result in the formula with the lowest error.

## Method
![Flow](https://user-images.githubusercontent.com/24732704/41862994-c34d9790-78a5-11e8-9c2d-8d3b78760d15.png)

* Generate a training and test set (by default 50/50).
* Run Monte Carlo simulation with n iterations, create a new model with random coefficients and test its prediction error. Store models with lower prediction errors than all previous models.
* Select and store the best model (lowest prediction error).
* Print (formula, error, confusion matrix) and plot (linear classification and reduction of error per iteration).

## Getting started
### Running in an environment
It's recommended to run the script in a contained Conda environment to ensure the installation of all dependencies, the required steps are listed below:

#### Install Miniconda
`$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`

`$ bash Miniconda3-latest-Linux-x86_64.sh`
#### Create and activate the environment
`$ conda env create --name {environment-name} --file environment.yml`

`$ source activate {environment-name}`
#### Run the script
`$ python monte_carlo_linear_classifier.py`

`$ source deactivate`

### Running as stand-alone
Alternatively, to run the script as a stand-alone, make sure the following dependencies are included:
* Python version 3
* Pip
* Matplotlib
* Numpy
* Pandas
* Seaborn

### Editing configurations
There are a couple of configurations that can be edited:

| Variable                  | Effect                                                                     |
| :------------------------ | :------------------------------------------------------------------------- |
| n_iterations (int)        | Changes the number of iterates, can significantly slow down your system!   |
| random_min_range (int)    | Changes the lower limit for coefficients                                   |
| random_max_range (int)    | Changes the upper limit for coefficients                                   |
| save_image (boolean)      | Enables saving output plots                                                |
| training_set_size (float) | Changes the percentage of data used as training set, between 0.0 and 1.0   |

## Credits
Originally coded as an assignment for a class on data mining. 
