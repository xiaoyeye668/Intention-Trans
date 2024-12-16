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

from modelling_functions import (
    normalize, 
    masked_mean,
    get_data
)


def lognormalize(x, axis):
    return x - pm.logsumexp(x, axis=axis, keepdims=True)


def normalize(x, axis):
    return x / x.sum(axis=axis, keepdims=True)
    

def yoon_S1(values, alpha, costs, L, phi, lib='np'):

    if lib == 'at':
        lib = at
    else:
        lib = np
    
    # dimensions (utterance, state)
    L0 = normalize(L,1)

    # expected value given each utterance
    # Shape (utterance)
    exp_values = lib.mean(
        values*L0,
        axis=1
    )
        
    # using a grid approximation for phi
    #phis = np.linspace(0,1,phi_grid_n)

    # p(u | s, phi)
    # where phi is essentially 
    # a politeness weight
    # Dimensions (phi, utterance, state)
    # NOTE: This is the same for all goal conditions
    S1 = normalize(
        lib.exp(alpha*(
            # informational component
            # (value of phi, utterance, state)
              phi[:,None,None]*lib.log(L0)
            # social component
            # (value of phi, utterance, 1)
            + (1-phi)[:,None,None]*exp_values[:,None]
            # (utterance, 1)
            - costs[:,None]
        )),
        # normalize by utterance
        axis=1
    )
    
    return S1


def yoon_utilities(S1, values):        
    # Prob of state and phi given utterance
    # Dimensions (phi, utterance, state)
    print('<<<<S1',S1.eval().shape)
    L1_s_phi_given_w = normalize(
        S1,
        (0,2)
    )    
    # grid-marginalize over phi
    #L1_s_given_w = L1_s_phi_given_w_grid.sum(0)
    #L1_s_given_w = L1_s_phi_given_w_grid[phi]
    # informativity of utterances given state
    # with utterances produced by L1
    # Shape (utterance, state)
    u_inf = pm.math.log(L1_s_phi_given_w)

          
    #print("U inf: ", u_inf.eval().shape)   (10,5)
    #print("U soc: ", u_soc.eval().shape)   (10,1)
    #print("U pres: ", u_pres.eval().shape) (3,10)

    # shape (utility component, goal condition, utterance, state)
    utilities = u_inf
    print('<<<<< utilities',L1_s_phi_given_w.eval().shape,utilities.eval().shape)
    return utilities


#def yoon_likelihood(alpha, values, omega, costs, phi, L, lib='np'):
def yoon_likelihood(alpha, values, costs, phi, L, lib='np'):
    S1 = yoon_S1(values, alpha, costs, L, phi, lib)
    utilities = yoon_utilities(S1, values) #shape:(3,3,10,5)(2,3,10,5)
    
    # print("omega: ", (
    #     omega
    # ).eval())

    # shape (goal condition, utterance, state)
    util_total = (
        # shape (goal condition, utterance, state)
        utilities
        # sum weighted utility components together
        # (utterance, 1)
        - costs[:,None]
    )

    # print("util_total: ", util_total.eval())
    
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

        # >= 0 
        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=20
        ) 
        #print('<<<<< alpha',alpha.eval())
        beta = pm.Uniform(
            'beta',
            lower=0,
            upper=10
        )
        values = beta*np.array([0, 1, 2, 3, 4])

        # goal weights
        # [informational, prosocial, presentational]
        # A triplet of goal weights for each goal condition!
        # Shape: (goal condition, utility component)
        #omega = pm.Uniform(
        #    'omega',
        #    lower=0,
        #    upper=1,
        #    shape=(3,)
        #)
        #print('<<<< omega',omega.eval())
        #phi_grid_n = 100

        # politeness weight
        # One for each goal condition!
        # Shape: (goal condition)
        #phi = pm.Categorical(
        #    'phi',
        #    p=np.ones(phi_grid_n)/phi_grid_n,
        #    shape=(3)
        #)
        phi = pm.Uniform(
            'phi',
            lower=0,
            upper=1,
            shape=(3,)
        )
        print('<<< phi ',phi.eval(),phi.eval().shape)
        # Shape (condition, utterance, state)
        S2 = yoon_likelihood(
            alpha, 
            values, 
            costs, 
            phi,
            L,
            lib='at'
        )
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
        start = pm.find_MAP()
        yoon_trace = pm.sample(
            draws=10000,
            tune=20000,
            chains=4,
            cores=8,
            initvals=start
        )
        ppc = pm.sample_posterior_predictive(yoon_trace)    
    az.to_netcdf(
        yoon_trace, 
        'traces/2017model_S2InfOnly_1wTune2w_LS2map_phiUni.cdf'
    )
    az.to_netcdf(ppc, 'ppcs/ppc_2017model_S2InfOnly_1wTune2w_LS2map_phiUni.cdf')
    summary = az.summary(yoon_trace,round_to=3)
    print(summary)