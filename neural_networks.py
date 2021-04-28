import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple


def plot_output(df:pd.DataFrame, disc_var:str, cont_var:str, disc_var_values:List[str], title:str=None,
                legend_values:str=None, xlabel:str=None, predict:str='prediction_', filename=None) -> None:
    '''
    Plots the output predictions of each neural network versus a continuous variable, 
    grouped by a discrete variable (e.g. age/gender vs the output).
    :param df: Pandas DataFrame, the dataset, also containing the predictions
    :param disc_var: String, the name of the discrete variable
    :param cont_var: String, the name of the continuous variable
    :param disc_var: List of strings, the name of the values of the discrete variable
    :param disc_var: List of strings, the values used in the legend (defaults to the 
                     name of the values of the discrete variable)
    :param predict: String, prefix name of column containing the predictions
    :return 
    '''
    if not predict+str(2) in df.columns: 
        f, axes = plt.subplots(1, 1, sharey=False, figsize=(5, 4))
        axes = [axes]
    else:
        f, axes = plt.subplots(1, 3, sharey=False, figsize=(15, 4))
        
    colors = ['-', '--']

    for idx, ax in enumerate(axes):
        for disc_id, disc_var_val in enumerate(disc_var_values):
            ax.plot(df[df[disc_var]==disc_var_val].groupby(cont_var).mean()[predict+str(idx+1)], colors[disc_id])
        ax.set(xlabel=xlabel if xlabel else cont_var, 
               title= title if title else str(idx+1) + ' layer',
               ylim=(-0.05,1.05)
              )
        if not legend_values: legend_values = disc_var_values
        ax.legend(legend_values)
    axes[0].set(ylabel= 'Output')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()