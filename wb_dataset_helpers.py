import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler

import copy
import numpy as np


# List of static variables

# Condition 1
THRESHOLD_AGE_WOMAN = 60
THRESHOLD_AGE_MAN = 65
MAX_AGE = 100
MIN_AGE = 0

# Condition 2
THRESOLD_PAID_CONTRIBUTIONS = 4

# Condition 5
MIN_CAPITAL_RESOURCES = 0
MAX_CAPITAL_RESOURCES = 10000 #5000
THRESHOLD_CAPITAL_RESOURCES = 3000

# Condition 
MIN_DISTANCE = 0
MAX_DISTANCE = 100
THRESHOLD_DISTANCE = 50

#Step size for patient-distance and age-gender datasets
STEP_SIZE = 5

def map_vars(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Map strings and booleans to integers
    :param df: Pandas DataFrame, the dataset
    :return: Pandas DataFrame, the dataset with mapped values
    '''
    # Map strings and Booleans to numbers
    mymap = {False:0, True:1, 'no':0, 'yes':1, 'm':0, 'f':1, 'in':0, 'out':1}
    return df.applymap(lambda s: mymap.get(s) if s in mymap else s)

def create_scaler(df:pd.DataFrame) -> MinMaxScaler:
    '''
    Creates a minmaxscaler based on a dataset
    :param df: the dataset to base the minmaxscaler on
    :return: the minmaxscaler
    '''
    df.pop('eligible')
    scaler = MinMaxScaler()
    scaler.fit(map_vars(df))
    return scaler

def preprocess(df:pd.DataFrame, scaler:MinMaxScaler, normalize:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Preprocesses the dataset into preprocessed dataset and labels
    :param df: Pandas DataFrame, the dataset
    :param normalize: Boolean, whether to normalize the data using a minmax scaler
    :return: Tuple of the preprocessed data as 2D numpy array, 
        and the labels as a 1D numpy array
    '''   

    # Copy the dataframe
    df = copy.deepcopy(df)
    
    # Get the label and remove it from the data
    y = df.pop('eligible').values.astype(int)

    # Map strings and Booleans to numbers
    df = map_vars(df)
    
    # Normalize the values between 0-1
    if normalize:
        int_vars = ['gender', 'is_spouse', 'is_absent', 'patient_type'] + ['paid_contribution_'+str(idx) for idx in range(1,6)]
        int_vars = [var for var in int_vars if var in df.columns]
        X = pd.DataFrame(scaler.transform(df), columns=df.columns)
        for col in int_vars:
            X[col] = X[col].astype(int)
    else:
        X = df
    return X.values, y

def analyse_instance(instance: pd.Series, return_rules:bool=False) -> Tuple[bool, Dict[int, bool]]:
    '''
    Analyses an instances.
    :param instance: Pandas series, an instance from the dataset
    :param return rules: Boolean, whether to return the rules 
    :return: Tuple of two values:
        Boolean, whether the instances is eligible
        Rules: Dictionary, for each rule whether it is satisfied by the instance
    '''
    rules = {}
    
    # Rule 1
    if instance['gender'] == 'm':
        rules['rule_1'] = True if instance['age'] >= THRESHOLD_AGE_MAN else False
    else:
        rules['rule_1'] = True if instance['age'] >= THRESHOLD_AGE_WOMAN else False
    
    # Rule 2
    rules['rule_2'] = [instance['paid_contribution_'+str(n)] for n in range(1,6)].count('yes') >= THRESOLD_PAID_CONTRIBUTIONS
    
    # Rule 3
    rules['rule_3'] = instance['is_spouse']

    # Rule 4
    rules['rule_4'] = not instance['is_absent']
    
    # Rule 5
    rules['rule_5'] = instance['capital_resources'] <= THRESHOLD_CAPITAL_RESOURCES
    
    # Rule 6
    if instance['patient_type'] == 'in':
        rules['rule_6'] = True if instance['distance_to_hospital'] < THRESHOLD_DISTANCE else False
    else:
        rules['rule_6'] = True if instance['distance_to_hospital'] >= THRESHOLD_DISTANCE else False
    
    # Count the number of failed rules
    rules['num_failed_rules'] = list(rules.values()).count(False)
    
    if return_rules:
        return rules['num_failed_rules']==0, rules
    return rules['num_failed_rules']==0

def analyse_db(db:pd.DataFrame, return_with_original:bool=False, verbose=True, analysis_function=analyse_instance) -> pd.DataFrame:
    '''
    Analyses a dataset to see which instances fail on which rules and
    whether there are mistakes in the dataset.
    :param db: The dataset
    :param return_with_original: Whether to return the analysis along with the original db
    :param analysis_function: the function used to analyse the instance
    :return: DataFrame with 
    '''
    all_rules = []
    all_outcomes = []
    
    if verbose: print('Dataset with', len(db), 'instances with', len(db.columns), 'columns')
    
    eligible_instances = db[db['eligible']==True]
    if verbose: print(len(eligible_instances)/len(db), 'of the instances are inelgible', 
          '(' + str(len(eligible_instances)) + '/' + str(len(db)) + ')')
    
    # Analyse each instance in the dataset
    for idx, instance in db.iterrows():
        outcome, rules = analysis_function(instance, return_rules=True)
        all_outcomes.append(outcome)
        all_rules.append(rules)
    
    # Check whether there are any mistakes in the dataset
    checks = [outcome==eligibility 
              for outcome, eligibility in zip(all_outcomes, db['eligible'])]
    if verbose: print('There are ', checks.count(False), 'mistakes in the dataset.')
    
    # Create a dataframe from the analysis
    analysis = pd.DataFrame(all_rules)
    analysis['correct'] = checks
    
    # Print number of false rules if ineligible
    mean_false_rules = analysis[db['eligible']==False]['num_failed_rules'].mean()
    if verbose: print('On average', mean_false_rules, 'condititions are unsatisfied if ineligible')
    
    # Concatenate the database with the analysis if specified
    if return_with_original:
        return pd.concat([db, analysis], axis=1)
    
    return analysis

def plot_hists(df:pd.DataFrame, steps=1) -> None:
    '''
    Plots histograms of all variables of the datasets
    :param df: The dataset
    '''
    f, axes = plt.subplots(2, 3, sharey=False, figsize=(15, 10))
    f.delaxes(axes[0,2])
    axes[0][0].hist(df['age'], bins=int((max(df['age'])-min(df['age']))/steps)+1)
    axes[0][1].hist(df['distance_to_hospital'], bins=int((max(df['distance_to_hospital'])-min(df['distance_to_hospital']))/steps)+1)
    axes[1][0].bar(train_df['gender'].value_counts().index, train_df['gender'].value_counts().values)
    axes[1][1].bar(train_df['patient_type'].value_counts().index, train_df['patient_type'].value_counts().values)
    axes[1][2].bar(train_df['eligible'].apply(str).value_counts().index, train_df['eligible'].apply(str).value_counts().values)
    for coor, title in zip([(0,0), (0,1), (1,2), (1,0), (1,1)], ['age', 'distance', 'eligible', 'gender', 'patient_type']):
        axes[coor[0]][coor[1]].set_title(title)
    plt.plot()


### Smaller domain
def analyse_instance_small(instance: pd.Series, return_rules:bool=False) -> Tuple[bool, Dict[int, bool]]:
    '''
    Analyses an instances.
    :param instance: Pandas series, an instance from the dataset
    :param return rules: Boolean, whether to return the rules 
    :return: Tuple of two values:
        Boolean, whether the instances is eligible
        Rules: Dictionary, for each rule whether it is satisfied by the instance
    '''
    rules = {}
    
    # Rule 1
    if instance['gender'] == 'm':
        rules['rule_1'] = True if instance['age'] >= THRESHOLD_AGE_MAN else False
    else:
        rules['rule_1'] = True if instance['age'] >= THRESHOLD_AGE_WOMAN else False
       
    # Rule 6
    if instance['patient_type'] == 'in':
        rules['rule_6'] = True if instance['distance_to_hospital'] < THRESHOLD_DISTANCE else False
    else:
        rules['rule_6'] = True if instance['distance_to_hospital'] >= THRESHOLD_DISTANCE else False
    
    # Count the number of failed rules
    rules['num_failed_rules'] = list(rules.values()).count(False)
    
    if return_rules:
        return rules['num_failed_rules']==0, rules
    return rules['num_failed_rules']==0
