import math

import pandas as pd
import numpy as np

from types import SimpleNamespace
from typing import Tuple

def test_instance(instance:dict) -> bool:
    '''
    Returns True if the instance has as a duty to repair damages (dut), False otherwise
    :param instance: the instance to be tested
    :return: Boolean, the value of the dut
    '''
    
    # Intialize the variables from the instance into a namespace
    n = SimpleNamespace(**instance)
    
    # Unlawful check
    n.unl = n.vun or (n.vst and not n.jus) or (n.vrt and not n.jus)
    
    # Imputed check
    n.imp = n.ico or n.ila or n.ift    
    
    return (n.dmg and n.unl and n.imp and n.cau) and not (n.vst and not n.prp)

def to_binary(integer:int, length:int=10) -> str:
    '''
    Converts an integer to binary representating of a certain length
    :param integer: The integer that will be converted to binary
    :param length: Int, the length that the binary output string will have
    :return: Binary output string
    '''
    binary = "{0:b}".format(integer)
    padding = ''.join('0' for _ in range(0, length-len(binary)))
    return padding+binary

def generate_unique_dataset() -> pd.DataFrame:
    '''
    Generates a dataset that contains all unique combination of variables and values.
    :return: Pandas Dataframe, the unique dataset
    '''
    
    # Generate all possible value combinations
    variables = ['dmg','cau', 'vrt', 'vst', 'vun', 'jus', 'ift', 'ila', 'ico', 'prp']
    potential_values = [to_binary(idx, length=len(variables)) for idx in range(0, 2**len(variables))]
    dataset = [{var:int(val[idx]) for idx, var in enumerate(variables)}
           for val in potential_values]
    
    # Provide the 'dut' label to each instance
    for instance in dataset:
        instance['dut'] = int(test_instance(instance))
    
    return pd.DataFrame(dataset)

def generate_dataset(db_size:int, false_ratio:float=0.5) -> pd.DataFrame:
    '''
    Generates a dataset with a predefined size and distribution.
    :param db_size: The number of instances in the dataset
    :param false_ratio: Ratio of how many instances have a false label
    :return: Pandas Dataframe, the dataset
    '''
   
    # Generate all possibile unique instances
    unique_dataset = generate_unique_dataset()
    np.random.shuffle(unique_dataset.values)
    unique_true_instances = list(unique_dataset[unique_dataset['dut']==True].T.to_dict().values())
    unique_false_instances = list(unique_dataset[unique_dataset['dut']==False].T.to_dict().values())
    
    dataset = []

    #Create instances with True 'dut' labels
    for n in range(0, math.floor(db_size * (1 - false_ratio))):
        dataset.append(unique_true_instances[n%len(unique_true_instances)])
        
    #Create instances with False 'dut' labels
    for n in range(0, math.ceil(db_size * false_ratio)):
        dataset.append(unique_false_instances[n%len(unique_false_instances)])
    
    return pd.DataFrame(dataset)

def preprocess(dataset:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Splits the dataset into data and labels for machine learning.
    :param dataset: Pandas Dataframe, the dataset
    :return: X, y as numpy arrays
    '''
    X = dataset.loc[:, dataset.columns != 'dut'].values
    y = dataset['dut'].values
    return X, y


def is_unlawful(instance:pd.Series) -> bool:
    '''
    Returns True if the act is unlawful
    :param instance: an instance from the dataset
    :return: Boolean, whether the act is unlawful
    '''
    return bool(instance['vun'] or (instance['vst'] and 
            not instance['jus']) or (instance['vrt'] and not instance['jus']))

def is_imputable(instance:pd.Series) -> bool:
    '''
    Returns True if the act can be imputed 
    :param instance: an instance from the dataset
    :return: Boolean, whether the act is imputable
    '''
    return bool(instance['ico'] or instance['ila'] or instance['ift'])


def generate_unlawful_dataset() -> pd.DataFrame:
    '''
    Generates a dataset in which all conditions are satisfied except unlawfulness.
    :return: Pandas DataFrame, the new dataset 
    '''
    
    # Generate all possibile unique instances
    df = generate_unique_dataset()
    
    # Remove instances with violations and without purpose (vst and not prp)
    df.drop(df.loc[(df.vst==1) & (df.prp==0)].index, inplace=True)
    
    # Remove instances without damage (dmg)
    df = df[df.dmg == 1]
    
    # Remove instances without impudence (imp)
    df.drop([index for index, row in df.iterrows() if not is_imputable(row)], inplace=True)
    
    # Remove intances without cause (cau)
    df = df[df.cau == 1]

    # Return the dataset
    return df

def generate_impudence_dataset() -> pd.DataFrame:
    '''
    Generates a dataset in which all conditions are satisfied except impudence.
    :return: Pandas DataFrame, the new dataset 
    '''
    
    # Generate all possibile unique instances
    df = generate_unique_dataset()
    
    # Remove instances with violations and without purpose (vst and not prp)
    df.drop(df.loc[(df.vst==1) & (df.prp==0)].index, inplace=True)
    
    # Remove instances without damage (dmg)
    df = df[df.dmg == 1]
    
    # Remove instances without unlawfulness (unl)
    df.drop([index for index, row in df.iterrows() if not is_unlawful(row)], inplace=True)
    
    # Remove intances without cause (cau)
    df = df[df.cau == 1]

    # Return the dataset
    return df