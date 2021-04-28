import copy
import numpy as np
import pandas as pd

from random import choice, randint, shuffle
from typing import Dict, List, Tuple

# Condition 1
THRESHOLD_AGE_WOMAN = 60 
THRESHOLD_AGE_MAN = 65
MAX_AGE = 100
MIN_AGE = 0

# Condition 2
MIN_DISTANCE = 0
MAX_DISTANCE = 100
THRESHOLD_DISTANCE = 50

# Steps used in age-gender/patient-distance datasets
STEP_SIZE = 5

def is_eligible(instance:pd.Series) -> bool:
    '''
    Returns true if the instance satisfies the conditions
    Return: True if the instance satisfies the conditions, false otherwise
    '''
    # Condition 1
    if (instance['gender'] == 'f' and instance['age'] < THRESHOLD_AGE_WOMAN):
        return False
    if (instance['gender'] == 'm' and instance['age'] < THRESHOLD_AGE_MAN):
        return False
    
    # Condition 2
    if (instance['patient_type'] == 'in' and instance['distance_to_hospital'] >= THRESHOLD_DISTANCE):
        return False
    if (instance['patient_type'] == 'out' and instance['distance_to_hospital'] < THRESHOLD_DISTANCE):
        return False
    return True

def create_unique_dataset() -> pd.DataFrame:
    '''
    Creates a dataset that contains all possible unique instances.
    :return: Pandas Dataframe, the dataset
    '''

    # Iterate through all combinations
    db = [{'age': age,
           'gender': gender,
           'patient_type': patient_type,
           'distance_to_hospital': distance}
        for age in range(MIN_AGE, MAX_AGE+1)
        for gender in ['m', 'f']
        for distance in range(MIN_DISTANCE, MAX_DISTANCE+1)
        for patient_type in ['in', 'out']]

    # Add eligibility labels
    for instance in db:
        instance['eligible'] = is_eligible(instance)
    
    return pd.DataFrame(db)

def create_age_gender_dataset() -> pd.DataFrame:
    '''
    Creates a dataset that contains all possible combinations of age and gender,
    where the patient-distance condition is always satisfied.
    :return: Pandas Dataframe, the dataset
    '''
    
    db = []
    
    # Iterate through all combinations of age/gender
    ag_combinations = [(age, gender) 
                       for gender in ['m', 'f']
                       for age in range(MIN_AGE, MAX_AGE+STEP_SIZE, STEP_SIZE)]
    
    # Select patient-type and distance such that condition 2 is satisfied
    pd_combinations = ([('in', distance) for distance in range(MIN_DISTANCE, THRESHOLD_DISTANCE)] +
                       [('out', distance) for distance in range(THRESHOLD_DISTANCE, MAX_DISTANCE+1)])
    
    for age, gender in ag_combinations:
        for patient_type, distance in pd_combinations:        
            # Create instance, add eligibility label and append to db
            instance = {
                'age': age,
                'gender': gender,
                'patient_type': patient_type,
                'distance_to_hospital': distance
            }
            instance['eligible'] = is_eligible(instance)
            db.append(instance)
    
    return pd.DataFrame(db)

def create_patient_distance_dataset() -> pd.DataFrame:
    '''
    Creates a dataset that contains all possible combinations of patient-typ and distance,
    where the age-gender condition is always satisfied.
    :return: Pandas Dataframe, the dataset
    '''
    
    db = []
    
    # Iterate through all combinations of patient-type/distance
    pd_combinations =  [(patient_type, distance) 
                       for patient_type in ['in', 'out']
                       for distance in range(MIN_DISTANCE, MAX_DISTANCE+STEP_SIZE, STEP_SIZE)]  
    
    # Select patient-type and distance such that condition 2 is satisfied
    ag_combinations = ([(age, 'm') for age in range(THRESHOLD_AGE_MAN, MAX_AGE+1)] +
                       [(age, 'f') for age in range(THRESHOLD_AGE_WOMAN, MAX_AGE+1)])  
    
    for age, gender in ag_combinations:
        for patient_type, distance in pd_combinations:        
            # Create instance, add eligibility label and append to db
            instance = {
                'age': age,
                'gender': gender,
                'patient_type': patient_type,
                'distance_to_hospital': distance
            }
            instance['eligible'] = is_eligible(instance)
            db.append(instance)
    
    return pd.DataFrame(db)

def create_instance(age, gender, patient_type, distance, eligible):
    return {
        'age': age,
        'gender': gender,
        'patient_type': patient_type,
        'distance_to_hospital': distance,
        'eligible': eligible,
    } 

def create_training_data(db_size:int=2400) -> pd.DataFrame:
    '''
    Creates a dataset that contains all possible combinations of patient-typ and distance,
    where the age-gender condition is always satisfied.
    :return: Pandas Dataframe, the dataset
    '''
    
    db = []

    # Eligible cases
    for _ in range(0, int(db_size/2)):
        gender, age = choice([
            ('m', randint(THRESHOLD_AGE_MAN, MAX_AGE)),
            ('f', randint(THRESHOLD_AGE_WOMAN, MAX_AGE)) 
        ])
        patient_type, distance = choice([
            ('in', randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1)),
            ('out', randint(THRESHOLD_DISTANCE, MAX_DISTANCE)) 
        ])
        db.append(create_instance(age, gender, patient_type, distance, True))
    
    # Ineligible cases that fail on condition 1
    for _ in range(0, int(db_size/4)):
        gender, age = choice([
            ('m', randint(MIN_AGE, THRESHOLD_AGE_MAN-1)),
            ('f', randint(MIN_AGE, THRESHOLD_AGE_WOMAN-1)) 
        ])
        patient_type, distance = choice([
            ('in', randint(MIN_DISTANCE, MAX_DISTANCE)),
            ('out', randint(MIN_DISTANCE, MAX_DISTANCE)) 
        ])
        db.append(create_instance(age, gender, patient_type, distance, False))
    
    # Ineligible cases that fail on condition 2
    for _ in range(0, int(db_size/4)):
        gender, age = choice([
            ('m', randint(MIN_AGE, MAX_AGE)),
            ('f', randint(MIN_AGE, MAX_AGE)) 
        ])
        patient_type, distance = choice([
            ('in', randint(THRESHOLD_DISTANCE, MAX_DISTANCE)),
            ('out', randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1)) 
        ])
        db.append(create_instance(age, gender, patient_type, distance, False))  
    
    return pd.DataFrame(db)

def create_training_data_B(db_size:int=2400) -> pd.DataFrame:
    '''
    Creates a dataset that contains all possible combinations of patient-typ and distance,
    where the age-gender condition is always satisfied. Instances fail on only a single condition
    :return: Pandas Dataframe, the dataset
    '''
    
    db = []

    # Eligible cases
    for _ in range(0, int(db_size/2)):
        gender, age = choice([
            ('m', randint(THRESHOLD_AGE_MAN, MAX_AGE)),
            ('f', randint(THRESHOLD_AGE_WOMAN, MAX_AGE)) 
        ])
        patient_type, distance = choice([
            ('in', randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1)),
            ('out', randint(THRESHOLD_DISTANCE, MAX_DISTANCE)) 
        ])
        db.append(create_instance(age, gender, patient_type, distance, True))
    
    # Ineligible cases that fail on condition 1
    for _ in range(0, int(db_size/4)):
        gender, age = choice([
            ('m', randint(MIN_AGE, THRESHOLD_AGE_MAN-1)),
            ('f', randint(MIN_AGE, THRESHOLD_AGE_WOMAN-1)) 
        ])
        patient_type, distance = choice([
            ('in', randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1)),
            ('out', randint(THRESHOLD_DISTANCE, MAX_DISTANCE)) 
        ])
        db.append(create_instance(age, gender, patient_type, distance, False))
    
    # Ineligible cases that fail on condition 2
    for _ in range(0, int(db_size/4)):
        gender, age = choice([
            ('m', randint(THRESHOLD_AGE_MAN, MAX_AGE)),
            ('f', randint(THRESHOLD_AGE_WOMAN, MAX_AGE)) 
        ])
        patient_type, distance = choice([
            ('in', randint(THRESHOLD_DISTANCE, MAX_DISTANCE)),
            ('out', randint(MIN_DISTANCE, THRESHOLD_DISTANCE-1)) 
        ])
        db.append(create_instance(age, gender, patient_type, distance, False))  
    
    return pd.DataFrame(db)


