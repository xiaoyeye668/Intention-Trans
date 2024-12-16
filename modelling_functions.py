import numpy as np
import seaborn as sns
import pandas as pd

import pymc as pm
try:
    import aesara
    import aesara.tensor as at
except ModuleNotFoundError:
    # If pymc v5 is installed
    import pytensor as aesara
    import pytensor.tensor as at
import arviz as az


def normalize(arr, axis):
    return arr / np.sum(arr, axis, keepdims=True)


def masked_mean(arr, mask):
    """
    Calculate the (row)mean of only those values
    in arr where mask is true
    """
    
    arr, mask = np.broadcast_arrays(
        arr, mask
    )
    
    return np.ma.masked_where(
        np.logical_not(mask), 
        arr
    ).mean(1).data.reshape(-1,1)

'''
subid	item	goal	state	positivity	utterance	utterance_enroll	positivity_ori	utterance_ori
sub1	1	informative	1	no_neg	terrible	yes_terrible	算是	糟糕的

subid	item	state	positivity	utterance	positivity_ori	utterance_ori	judgment
sub1	1	4	neg	good	不是	上乘的	0
'''
def get_data(path="./speaker_production_65subs.csv",
             path_meaning='./literal_semantics.csv'):

    dt = pd.read_csv(path)

    us = [
        'terrible',
        'bad',
        'okay',
        'good',
        'perfect',
        'not terrible',
        'not bad',
        'not okay',
        'not good',
        'not perfect'
    ]

    ids, _ = dt.subid.factorize()
    dt.loc[:,'id'] = ids
    dt = dt.drop(columns='subid')
    dt.loc[:,'state'] = dt['state'].astype(int)-1
    #dt.loc[:,'state'] = dt['state'].astype(int)

    dt.loc[:,'utterance_full'] = np.where(
        dt['positivity']=='no_neg',
        dt.utterance,
        'not ' + dt.utterance
    )
    utt_i = {u:i for i,u in enumerate(us)}
    dt.loc[:,'utterance_index'] = dt.utterance_full.replace(utt_i)

    goal_id, goals = dt.goal.factorize()
    dt.loc[:,'goal_id'] = goal_id

    dt_meaning = pd.read_csv(path_meaning)
    dt_meaning.loc[:,'state'] = dt_meaning['state'].astype(int)-1
    #dt_meaning.loc[:,'state'] = dt_meaning['state'].astype(int)
    
    dt_meaning.loc[:,'neg'] = np.where(
        (dt_meaning.positivity == 'no_neg'),
        '',
        'not '  
    )
    
    dt_meaning.loc[:,'utterance_index'] = (
        dt_meaning['neg'] 
        + dt_meaning['utterance']
    ).replace(utt_i)
    
    return dt, utt_i, goal_id, goals, dt_meaning

#dt, utt_i, goal_id, goals, dt_meaning = get_data()
#print(dt, dt_meaning)

def get_shuffle_data(path="./speaker_production_65subs.csv",
             path_meaning='./literal_semantics.csv'):

    dt = pd.read_csv(path)

    us = [
        'terrible',
        'bad',
        'okay',
        'good',
        'perfect',
        'not terrible',
        'not bad',
        'not okay',
        'not good',
        'not perfect'
    ]
    
    ids, _ = dt.subid.factorize()
    dt.loc[:,'id'] = ids
    dt = dt.drop(columns='subid')
    dt.loc[:,'state'] = dt['state'].astype(int)-1
    #dt.loc[:,'state'] = dt['state'].astype(int)

    dt.loc[:,'utterance_full'] = np.where(
        dt['positivity']=='no_neg',
        dt.utterance,
        'not ' + dt.utterance
    )
    utt_i = {u:i for i,u in enumerate(us)}
    dt.loc[:,'utterance_index'] = dt.utterance_full.replace(utt_i)

    goal_id, goals = dt.goal.factorize()
    dt.loc[:,'goal_id'] = goal_id
    dt = dt.sample(frac=1, random_state=123)

    dt_meaning = pd.read_csv(path_meaning)
    dt_meaning.loc[:,'state'] = dt_meaning['state'].astype(int)-1
    #dt_meaning.loc[:,'state'] = dt_meaning['state'].astype(int)
    
    dt_meaning.loc[:,'neg'] = np.where(
        (dt_meaning.positivity == 'no_neg'),
        '',
        'not '  
    )
    
    dt_meaning.loc[:,'utterance_index'] = (
        dt_meaning['neg'] 
        + dt_meaning['utterance']
    ).replace(utt_i)
    
    return dt, utt_i, goal_id, goals, dt_meaning
