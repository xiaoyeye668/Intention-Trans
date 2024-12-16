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
    utilities = utility_for_S1(costs, L) #shape:(10,5)
    print('<<<<<< utilities ',utilities.eval().shape)
    '''
    # shape (goal condition, utterance, state)
    util_total = (
        # dims: (goal condition, 1, 1)
        omega[:,None,None] 
        # dims: (utterance, state)
        * utilities
        # (utterance, 1)
        - costs[:,None]
    )
    '''
    # prob of utterance given state
    # Shape: (utterance, state)
    S1 = normalize(
        pm.math.exp(alpha * utilities), 
        axis=0
    )
    
    return S1

def S2_utilities(S1, phi, values, costs):        
    # Prob of state and phi given utterance
    # Dimensions (utterance, state)
    L1_s_given_w = normalize(
        S1,
        axis=1
    )    
    
    # informativity of utterances given state
    # with utterances produced by L1
    # Shape (utterance, state)
    u_inf = pm.math.log(L1_s_given_w)

    # expected (value of state)
    # for each utterance as produced by L1
    # NOTE: Same for all goal conditions
    u_soc = at.mean(
        values*L1_s_given_w,
        1
    )
    print("U inf: ", u_inf.eval().shape)
    print("U soc: ", u_soc.eval().shape)

    # shape (phi, utterance, state)
    util_total = (
            # informational component
            # (value of phi, utterance, state)
              phi.T[0,:,None,None]*u_inf
            # social component
            # (value of phi, utterance, 1)
            + phi.T[1,:,None,None]*u_soc[:,None]
            # (utterance, 1)
            - costs[:,None]
        )
    
    return util_total


def S2_likelihood(alpha, values, costs, phi, L, lib='np'):

    S1 = S1_likelihood(alpha, costs, L, lib)

    util_total = S2_utilities(S1, phi, values, costs)

    print("<<<<< util_total: ", util_total.eval().shape)
    
    # for each goal condition,
    # prob of utterance given state
    # Shape: (goal_condition, utterance, state)
    S2 = normalize(
        pm.math.exp(alpha*util_total), 
        1
    )
    
    return S2

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
            [9.937e-01, 5.681e-01, 2.172e-01, 2.760e-02, 1.000e-10],
            [6.596e-01, 9.032e-01, 3.296e-01, 1.113e-01, 1.517e-01],
            [7.160e-02, 2.683e-01, 7.393e-01, 4.520e-01, 3.716e-01],
            [1.000e-04, 1.150e-02, 3.036e-01, 9.497e-01, 9.386e-01],
            [1.000e-10, 1.000e-10, 3.000e-04, 2.912e-01, 9.882e-01],
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

        # >= 0 
        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=20,
        ) 
        beta = pm.Uniform(
            'beta',
            lower=0,
            upper=5
        )
        values = beta * np.array([0, 1, 2, 3, 4])

        phi = pm.Dirichlet(
            'phi',
            [1,1],
            shape=(3,2)
        )
        
        # Shape (condition, utterance, state)
        S2 = S2_likelihood(alpha, values, costs, phi, L, lib='at')
        print('<<<<<<<< S2', S2.eval().shape)
        # each combination of goal and state
        # should give a prob vector over utterances
        # print(S2.eval().sum((1)))
        #print('<<<<<<<<< L', L)
        #print('<<<<<<<< dt', dt)
        #print('<<<<<<<< L_observed ',L_observed.eval(),L_observed.eval().shape)
        p_production = S2[
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
        yoon_trace = pm.sample(
            draws=10000,
            tune=20000,
            chains=4,
            cores=8
        )
        ppc = pm.sample_posterior_predictive(yoon_trace)    
    az.to_netcdf(
        yoon_trace, 
        'traces/2020model_S2_1wTune2w_LS2map_S1noE.cdf'
    )
    az.to_netcdf(ppc, 'ppc_2020model_S2_1wTune2w_LS2map_S1noE.cdf')
    summary = az.summary(yoon_trace,round_to=3)
    print(summary)

    
