'''
AUTHOR: Koen Rademaker
FILE: classifier.py
VERSION: 1.0
DATE: 25/jun/2018
FUNCTION: Run Monte Carlo simulations to get the best linear classification model.
'''
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns

file_location = 'Data/classification.txt'
n_iterations = 50000
random_min_range = -30
random_max_range = 30
save_image = True
training_set_size = 0.5

class Classifier:
    a = 0
    b = 0
    c = 0
    error = 0
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

# Load data from file and store as dataframe
def load_data(file_location):
    dataframe = pd.read_table(file_location, sep='\t', header='infer')
    dataframe.loc[:,'Prediction'] = pd.Series(index=dataframe.index)
    dataframe.loc[:,'Error'] = pd.Series(index=dataframe.index)
    return dataframe

# Run Monte Carlo simulations to generate classification models
def monte_carlo(dataframe):
    improved_models = []
    for i in range(n_iterations):
        model = make_model(dataframe)
        n_improved_models = len(improved_models)
        if n_improved_models != 0:
            min_error = improved_models[n_improved_models-1].error
            if model.error < min_error:
                improved_models.append(model)
        else:
            improved_models.append(model)
    return improved_models

# Randomly generate an integer
def get_random_int():
    return np.random.randint(random_min_range, random_max_range)

# Make a classification model
def make_model(dataframe):
    result = test_model(Classifier(get_random_int(), get_random_int(), get_random_int()), dataframe)
    return result['Model']

# Test efficiency of classification model
def test_model(model, dataframe, restrict_error=False):
    for index, row in dataframe.iterrows():
        formula = model.a*row['Polarity']+model.b*row['Hydrophobicity']+model.c
        try:
            prediction = 1/(1+math.exp(-formula))
        except OverflowError:
            prediction = float('inf')
        dataframe.loc[index, 'Prediction'] = prediction
        dataframe.loc[index, 'Error'] = (row['Class'] - prediction)**2
    dataframe['Error'] = dataframe['Error'].abs()
    if restrict_error == False:
        model.error = dataframe.loc[:,'Error'].sum()
    return {'Model': model, 'Dataframe': dataframe}

# Get confusion matrix for a model
def get_confusion_matrix(model, dataframe):
    result = test_model(model, dataframe, True)['Dataframe']
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for index, row in result.iterrows():
        actual = result.loc[index, 'Class']
        predicted = result.loc[index, 'Prediction']
        if actual >= 0.5 and predicted >= 0.5:
            true_pos += 1
        if actual < 0.5 and predicted < 0.5:
            true_neg += 1
        if actual < 0.5 and predicted >= 0.5:
            false_pos += 1
        if actual >= 0.5 and predicted < 0.5:
            false_neg += 1
    confusion_matrix = pd.DataFrame({'0': [true_neg, false_pos], '1': [false_neg, true_pos]}, index=['0', '1'],columns=['0', '1'])
    confusion_matrix.index.name = 'Prediction'
    confusion_matrix.columns.name = '    Actual'
    return confusion_matrix

# Plot model and decision boundary
def plot_model(a, b, c, dataframe, save_image):
    sns.pairplot(x_vars=['Polarity'], y_vars=['Hydrophobicity'], data=dataframe, hue='Class', palette='bright', size=5)
    plt.plot(dataframe['Polarity'], (a*dataframe['Polarity'])+(b*dataframe['Polarity'])+c, c='black')
    plt.title('Classifier (y = '+str(a)+'*Polarity + '+str(b)+'*Hydrophobicity + '+str(c)+')', fontsize=12.5)
    if save_image:
        plt.savefig('Output/Classifier.svg', bbox_inches='tight')

# Plot reduction of model error as a result of simulations
def plot_simulations(models, save_image):
    plt.figure()
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    indices = []
    errors = []
    for i, model in enumerate(models):
        indices.append(i)
        errors.append(model.error)
    plt.tight_layout()
    plt.scatter(indices, errors)
    plt.title('Model error reduction ater '+str(len(models))+' improved iterations', fontsize=12.5)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Model error', fontsize=10)
    if save_image:
        plt.savefig('Output/Simulations.svg', bbox_inches='tight')

# Main function
def main():
    # Load data from file & generate training and test sets.
    file_df = load_data(file_location)
    training_set = file_df.sample(frac=training_set_size, random_state=10)
    test_set = file_df.drop(training_set.index)

    # Run Monte Carlo simulation and get the best model.
    monte_carlo_models = monte_carlo(training_set)
    best_model = monte_carlo_models[len(monte_carlo_models)-1]
    model_cm = get_confusion_matrix(best_model, test_set)

    # Report results of best model (formula & error, confusion matrix, visualization)
    print('Model:\ty = ',best_model.a,'*Polarity + ',best_model.b,'*Hydrophobicity + ',best_model.c, sep='')
    print('Error:\t', best_model.error, sep='')
    print(model_cm)
    plot_model(best_model.a, best_model.b, best_model.c, test_set, save_image)
    plot_simulations(monte_carlo_models, save_image)

main()
