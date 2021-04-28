import math
import pandas as pd
import numpy as np
import itertools as it

from random import choice, randint, shuffle
from typing import Dict, List, Tuple
from joblib import Parallel, delayed

from wb_dataset_helpers import *

def create_age_gender(satisfied: bool) -> (str, int):
    '''
    Generates random values for gender and age, 
    based on whether the rule "The person should be of 
    pensionable age (60 for a woman, 65 for a man)" should 
    satisfied or not.
    :param satisfied: Boolean, whether the rule should be satisfied
    :return: gender (str), age (int)
    '''
    gender = choice(['m', 'f'])
    threshold_age = THRESHOLD_AGE_WOMAN if gender == 'f' else THRESHOLD_AGE_MAN
    if satisfied:
        age = randint(threshold_age, MAX_AGE)
    else:
        age = randint(MIN_AGE, threshold_age-1)
    return age, gender

def create_paid_contributions(satisfied: bool) -> List[str]:
    '''
    Generates random values for the previously paid contributions based
    on whether the rule "The person should have paid contributions in four 
    out of the last five relevant contribution years" is satisfied or not.
    :param satisfied: Boolean, whether the rule should be satisfied
    :result: List of strings, yes and no for each of the last five contribution years
    '''
    if satisfied: 
        num_ones = randint(THRESOLD_PAID_CONTRIBUTIONS,5)
    else:
        num_ones = randint(0,THRESOLD_PAID_CONTRIBUTIONS-1)
    paid_contributions = ['yes' if x < num_ones else 'no' for x in range(0,5)]
    shuffle(paid_contributions)
    return paid_contributions

def create_type_distance(satisfied: bool, thresold:int=50) -> (str, int):
    '''
    Generates random values for patient type and distance, 
    based on whether the rule "If the relative is an in-patient 
    the hospital shouldbe within a certain distance: if an out-patient, 
    beyond that distance" is satisfied or not. 
    :param satisfied: Boolean, whether the rule should be satisfied
    :param thresold: The threshold to determine the 'certain distance'
    :return: type (str), distance (int)
    '''
    patient_type = choice(['in', 'out'])
    if patient_type == 'in':
        distance = randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1) if satisfied else randint(THRESHOLD_DISTANCE, MAX_DISTANCE)
    else:
        distance = randint(THRESHOLD_DISTANCE, MAX_DISTANCE) if satisfied else randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1)
    return patient_type, distance

def create_instance_single_fail(fail_rule:int=None) -> dict:
    '''
    Generates an instance in the dataset (information of a single person).
    If the instance in not eligible, it is because only one rule was not satisfied.
    :param fail_rule: The number of the rule on which this instance is not eligible
    :return: A dict with all of the information of the person
    '''
    age, gender = create_age_gender(fail_rule!=1)
    paid_contributions = create_paid_contributions(fail_rule!=2)
    patient_type, distance_to_hospital = create_type_distance(fail_rule!=6)
    instance = {'age': age,
                'gender': gender,
                'paid_contribution_1': paid_contributions[0],
                'paid_contribution_2': paid_contributions[1],
                'paid_contribution_3': paid_contributions[2],
                'paid_contribution_4': paid_contributions[3],
                'paid_contribution_5': paid_contributions[4],
                'is_spouse': False if fail_rule==3 else True,
                'is_absent': True if fail_rule==4 else False,
                'capital_resources': randint(THRESHOLD_CAPITAL_RESOURCES+1,MAX_CAPITAL_RESOURCES) if fail_rule==5 
                                     else randint(MIN_CAPITAL_RESOURCES, THRESHOLD_CAPITAL_RESOURCES),
                'patient_type': patient_type,
                'distance_to_hospital': distance_to_hospital,
                'eligible': fail_rule==None,
            }
    return instance

def create_instance(fail_rule:int=None) -> dict:
    '''
    Generates an instance in the dataset (information of a single person).
    If the instance in not eligible, it is because one rule in particular was not satisfied,
    though muliple rules can be unsatisfied.
    :param fail_rule: The number of the rule on which this instance is not eligible
    :return: A dict with all of the information of the person
    '''
    
    # If the instance is eligible
    if not fail_rule:
        return create_instance_single_fail(None)
    
    # Otherwise, generate an unsatisfied instance that can fail on multiple conditionss 
    age, gender = create_age_gender(fail_rule!=1)
    paid_contributions = create_paid_contributions(fail_rule!=2)
    patient_type, distance_to_hospital = create_type_distance(fail_rule!=6)
    instance = {'age': age if fail_rule==1 else randint(MIN_AGE,MAX_AGE),
                'gender': gender if fail_rule==1 else choice(['m', 'f']),
                'paid_contribution_1': paid_contributions[0] if fail_rule==2 else choice(['yes', 'no']),
                'paid_contribution_2': paid_contributions[1] if fail_rule==2 else choice(['yes', 'no']),
                'paid_contribution_3': paid_contributions[2] if fail_rule==2 else choice(['yes', 'no']),
                'paid_contribution_4': paid_contributions[3] if fail_rule==2 else choice(['yes', 'no']),
                'paid_contribution_5': paid_contributions[4] if fail_rule==2 else choice(['yes', 'no']),
                'is_spouse': False if fail_rule==3 else choice([True, False]),
                'is_absent': True if fail_rule==4 else choice([True, False]),
                'capital_resources': randint(THRESHOLD_CAPITAL_RESOURCES, MAX_CAPITAL_RESOURCES) if fail_rule==5 
                                     else randint(MIN_CAPITAL_RESOURCES, MAX_CAPITAL_RESOURCES),
                'patient_type': patient_type if fail_rule==6 else choice(['in', 'out']),
                'distance_to_hospital': distance_to_hospital if fail_rule==6 else randint(MIN_DISTANCE,MAX_DISTANCE),
                'eligible': False,
            }
    
    return instance

def add_noise(db: List[Dict], n_noise:int) -> List[Dict]:
    '''
    Adds noisy variables to all instances of the dataset
    :param db: the dataset (list of dictionaries)
    :param n_noise: the number of noise variables (int)
    :return: The new dataset with noise variables (list of dictionaries)
    '''

    return [{**instance, **{'noise_'+str(n+1): randint(0,100) 
                            for n in range(0, n_noise)}}
            for instance in db]

def create_dataset(db_size:int, instance_function, rule_number:int=None,
                   fail_ratio:int=0.5, n_noise:int=0, num_rules:int=6) -> pd.DataFrame:
    '''
    Creates a dataset (pandas DataFrame) of the personal information of eldery people.
    :param db_size: The number of instances in the dataset
    :param instance_function: The function used to generate instances
    :param rule_number: The number of the rule on which the instances should fail
    :param fail_ratio: Ratio of how many instances fail/are ineligible
    :param n_noise: The amount of noise variables that are added to the data
    :param num_rules: the number of rules, which is usually 6
    :return: The dataset as a pandas DataFrames
    '''
    db = []
    
    # The satisfied cases
    for n in range(0, math.floor(db_size * (1 - fail_ratio))):
        db.append(instance_function(None))
    
    # The unsatisfied cases
    for n in range(0, math.ceil(db_size * fail_ratio)):
        current_rule = rule_number if rule_number else n%num_rules + 1 
        db.append(instance_function(current_rule))
    
    # Add noise variables
    db = add_noise(db, n_noise)
    
    return pd.DataFrame(db)

def create_age_gender_dataset(db_size=1, n_noise=0, instance_function=create_instance_single_fail):
    '''
    Returns a dataset with instances in which all conditions are satisfied
    except for condition 1, which is generated across the full range.
    :param db_size: The number of times the full range is generated
    :param n_noise: The amount of noise variables that are added to the data
    :return: The dataset as a pandas DataFrames    
    '''
    
    db = []
    age_gender_combinations = [(age, gender) for gender in ['m', 'f']
                                             for age in range(MIN_AGE, MAX_AGE+STEP_SIZE, STEP_SIZE)]
    for idx in range(0, db_size):
        for age, gender in age_gender_combinations:
            eligible = ((gender=='m' and age >= THRESHOLD_AGE_MAN) or 
                        (gender=='f' and age >= THRESHOLD_AGE_WOMAN))
            instance = instance_function(None)
            instance.update({'age': age,
                             'gender': gender,
                             'eligible': eligible,
                            })
            db.append(instance)
    db = add_noise(db, n_noise)
    return pd.DataFrame(db)


def create_patient_distance_dataset(db_size=1, n_noise=0, instance_function=create_instance_single_fail):
    '''
    Returns a dataset with instances in which all conditions are satisfied
    except for condition 6, which is generated across the full range.
    :param db_size: The number of times the full range is generated
    :param n_noise: The amount of noise variables that are added to the data
    :return: The dataset as a pandas DataFrames    
    '''
    
    db = []
    type_distance_combinations = [(patient_type, distance) for patient_type in ['in', 'out']
                                  for distance in range(MIN_DISTANCE, MAX_DISTANCE+STEP_SIZE, STEP_SIZE)]
    for idx in range(0, db_size):
        for patient_type, distance in type_distance_combinations:
            eligible = ((patient_type=='in' and distance < THRESHOLD_DISTANCE) or 
                        (patient_type=='out' and distance >= THRESHOLD_DISTANCE))
            instance = instance_function(None)
            instance.update({'patient_type': patient_type,
                             'distance_to_hospital': distance,
                             'eligible': eligible,
                            })
            db.append(instance)
    db = add_noise(db, n_noise)
    return pd.DataFrame(db)


def is_eligible(instance:dict) -> bool:
    '''
    Returns true if the instance is eligible, false otherwise
    :param instance: instance in the dataset
    '''
    return analyse_instance(instance, return_rules=False)
