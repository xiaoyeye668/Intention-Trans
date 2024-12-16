import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
try:
    import aesara
    import aesara.tensor as at
except ModuleNotFoundError:
    import pytensor as aesara
    import pytensor.tensor as at
import arviz as az

from modelling_functions import get_data

def normalize(x, axis):
    return x / x.sum(axis=axis, keepdims=True)

def utility_for_S1(costs, L, lib='np'):
    if lib == 'at':
        lib = at
    else:
        lib = np
    # Shape (utterance, state)
    L0 = normalize(L,1)
    # expected value given each utterance

    utility = lib.log(L0) - costs[:,None]
    
    return utility    

def S1_likelihood(alpha, costs, L, lib='np'):
    # shape (goal condition, utterance, state)
    utilities = utility_for_S1(costs, L) #shape:(3,10,5)
    print('<<<<<< utilities ',utilities.eval().shape)

    # shape (goal condition, utterance, state)
    util_total = (
        # dims: (goal condition, 1, 1)
        alpha[:,None,None] 
        # dims: (utterance, state)
        * utilities

    )

    # for each goal condition,
    # prob of utterance given state
    # Shape: (goal_condition, utterance, state)
    S1 = normalize(
        pm.math.exp(util_total), 
        1
    )
    
    return S1

def factory_yoon_model(dt, dt_meaning):
    
    dt_meaning_pymc = (
        dt_meaning
        .groupby(['state', 'utterance_index'])
        ['judgment']
        .agg([np.sum, 'count'])
        .reset_index()
    )

    with pm.Model() as yoon_model:

        # literal semantic compatibility
        # shape: (utterances, states)
        L = np.array([
            [9.937e-01, 5.681e-01, 2.172e-01, 2.760e-02, 1.000e-5],
            [6.596e-01, 9.032e-01, 3.296e-01, 1.113e-01, 1.517e-01],
            [7.160e-02, 2.683e-01, 7.393e-01, 4.520e-01, 3.716e-01],
            [1.000e-04, 1.150e-02, 3.036e-01, 9.497e-01, 9.386e-01],
            [1.000e-5, 1.000e-5, 3.000e-04, 2.912e-01, 9.882e-01],
            [4.823e-01, 2.703e-01, 4.462e-01, 5.998e-01, 6.432e-01],
            [2.500e-01, 5.365e-01, 5.950e-01, 6.870e-01, 6.884e-01],
            [5.530e-01, 5.733e-01, 6.066e-01, 5.438e-01, 2.203e-01],
            [6.702e-01, 7.532e-01, 8.174e-01, 4.247e-01, 2.870e-02],
            [5.508e-01, 4.831e-01, 6.949e-01, 8.714e-01, 9.640e-02]
        ])
        '''
        L = pm.Uniform(
            'L',
            lower=0,
            upper=1,
            shape=(10,5)
        )
        #print(dt_meaning_pymc)
        #print(L.eval(),L.eval().shape)
        L_observed = pm.Binomial(
            'L_observed',
            n=dt_meaning_pymc['count'],
            p=L[
                dt_meaning_pymc['utterance_index'],
                dt_meaning_pymc['state']
            ],
            observed=dt_meaning_pymc['sum']
        )
        '''
        negative_cost = pm.Uniform(
            'c',
            lower=1,
            upper=10
        )

        costs = at.concatenate((
            at.ones(5),
            at.repeat(negative_cost,5)
        ))

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=100,
            shape=(3,)
        )
        
        # Shape (condition, utterance, state)
        S1 = S1_likelihood(alpha, costs, L, lib='at')
        print('<<<<<<<< S1', S1.eval().shape)
        # each combination of goal and state
        # should give a prob vector over utterances
        # print(S2.eval().sum((1)))
        #print('<<<<<<<<< L', L)
        #print('<<<<<<<< dt', dt)
        #print('<<<<<<<< L_observed ',L_observed.eval(),L_observed.eval().shape)
        p_production = S1[
            dt.goal_id,
            :,
            dt.state
        ]
        print('<<<<<< p_production ',p_production.eval().shape)
        pm.Categorical(
            "chosen",
            p_production,
            observed=dt.utterance_index.values,
            shape=len(dt)
        )
        
    return yoon_model


if __name__=='__main__':
    
    dt, utt_i, goal_id, goals, dt_meaning = get_data()
    
    yoon_model = factory_yoon_model(
        dt,
        dt_meaning
    )
    
    with yoon_model:
        start = pm.find_MAP()
        yoon_trace = pm.sample(
            draws=10000,
            tune=10000,
            chains=4,
            cores=8,
            initvals=start,
            target_accept=0.99
        )
        ppc = pm.sample_posterior_predictive(yoon_trace)    
    az.to_netcdf(
        yoon_trace, 
        'traces/S1_1wTune1w_LS2map_onlyInf_alpha.cdf'
    )
    az.to_netcdf(ppc, 'ppcs/ppc_S1_1wTune1w_LS2map_onlyInf_alpha.cdf')
    summary = az.summary(yoon_trace,round_to=3)
    print(summary)

    
