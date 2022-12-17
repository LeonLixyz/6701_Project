'''
NEB 

'''

import numpy as np
import copy
import os
np.seterr(all='raise')
import argparse
import csv
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime
import scipy
from scipy import stats
from scipy.special import gamma, digamma, loggamma


from LDA import compute_logsumexp, compute_elbo

'''
hyper parameters
'''
T = 32
N = 10
k = 0.01
lr = 0.1

# initialize the path
def init(theta_1, theta_2, N):
    LAMBDA_1, GAMMA_1, PHI_1 = theta_1
    LAMBDA_2, GAMMA_2, PHI_2 = theta_2

    interval = np.linspace(0,1,N+1)
  
    p_list = []

    for i in interval:
        NEW_LAMBDA = (LAMBDA_2 - LAMBDA_1) * i + LAMBDA_1
        NEW_GAMMA = (GAMMA_2 - GAMMA_1) * i + GAMMA_1
        NEW_PHI = [(Phi_2 - Phi_1) * i + Phi_1 for (Phi_1,Phi_2) in zip(PHI_1,PHI_2)]
        p_list.append([NEW_LAMBDA, NEW_GAMMA, NEW_PHI])

    return p_list

def dLdp(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C):
    predict_flag = False
    LAMBDA_t = copy.deepcopy(LAMBDA) # Shape: (C,V)
    GAMMA_t = copy.deepcopy(GAMMA) # Shape: (N,C)
    PHI_t = copy.deepcopy(PHI) # Shape: (N,n_words,C)

    N, V = train_articles.shape

    for i in tqdm(range(N)):
        article = train_articles[i]
        nonzero_idx = train_nonzero_idxs[i]

        # Fetch for PHI_ij update
        GAMMA_i_t = copy.deepcopy(GAMMA_t[i]) # C-vector

        # For each word in document
        corr_idx = 0

        # Iterate through each word with non-zero count on document
        for idx in nonzero_idx:
            log_PHI_ij = np.zeros((C,))

            for k in range(C):
                # Fetch for PHI_ij update
                LAMBDA_k_t = copy.deepcopy(LAMBDA_t[k]) # V-vector

                exponent = digamma(GAMMA_i_t[k]) - digamma(np.sum(GAMMA_i_t))
                exponent += digamma(LAMBDA_k_t[idx]) - digamma(np.sum(LAMBDA_k_t))
                log_PHI_ij[k] = exponent

            # Normalize using log-sum-exp trick
            PHI_ij = np.exp(log_PHI_ij - compute_logsumexp(log_PHI_ij))
            try:
                assert(np.abs(np.sum(PHI_ij) - 1) < 1e-6)
            except:
                raise AssertionError('phi_ij: {}, Sum: {}'.format(PHI_ij, np.sum(PHI_ij)))

            PHI_t[i][corr_idx] = PHI_ij
            corr_idx += 1

        # Check if number of updates match with number of words
        assert(corr_idx == len(nonzero_idx))

        # Update GAMMA_i
        GAMMA_i_t = np.zeros((C,)) + ALPHA

        for k in range(C):
            GAMMA_i_t[k] += np.sum(article[nonzero_idx] * PHI_t[i][:,k])

        GAMMA_t[i] = GAMMA_i_t

    if not predict_flag:
        # For each topic
        print('Updating LAMBDA')

        for k in tqdm(range(C)):
            LAMBDA_k_t = np.zeros((V,)) + ETA

            # For each document
            for i in range(N):
                article = train_articles[i]
                nonzero_idx = train_nonzero_idxs[i]

                # For each word in document
                corr_idx = 0

                for idx in nonzero_idx:
                    LAMBDA_k_t[idx] += article[idx] * PHI_t[i][corr_idx][k]
                    corr_idx +=1

                # Check if number of updates match with number of words
                assert(corr_idx == len(nonzero_idx))

            LAMBDA_t[k] = LAMBDA_k_t
    grad_LAMBDA = LAMBDA_t - LAMBDA
    grad_GAMMA = GAMMA_t - GAMMA
    grad_PHI = [phi_t - phi for (phi_t,phi) in zip(PHI_t, PHI)]

    return grad_LAMBDA, grad_GAMMA, grad_PHI

# Vertical Force and Horizontal Force
def F_L(i, p_list, train_articles, train_nonzero_idxs, C):

    LAMBDA, GAMMA, PHI = p_list[i]
    tau_Lam, tau_Gam, tau_Phi = tau(i, p_list, train_articles, train_nonzero_idxs, C)
    grad_LAMBDA, grad_GAMMA, grad_PHI = dLdp(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C)
    grad_dot_tau_Lam = np.multiply(grad_LAMBDA , tau_Lam)
    grad_dot_tau_Gam = np.multiply(grad_GAMMA , tau_Gam)
    grad_dot_tau_Phi = [np.multiply(grad_p ,tau_p) for (grad_p, tau_p) in zip (grad_PHI, tau_Phi)]
    F_L_Lam = -(grad_LAMBDA - grad_dot_tau_Lam*tau_Lam )
    F_L_Gam = -(grad_GAMMA - grad_dot_tau_Gam*tau_Gam )
    F_L_Phi = [-(grad_p - gdt_p*tau_p) for (grad_p, gdt_p, tau_p) in zip (grad_PHI, grad_dot_tau_Phi, tau_Phi)]
    return F_L_Lam, F_L_Gam, F_L_Phi
    #return -(dLdp(p_list[i]) - np.dot(dLdp(p_list[i]), tau(i, p_list)) *  tau(i, p_list))

def F_s(i, k, p_list):
    LAMBDA_1, GAMMA_1, PHI_1 = p_list[i]
    LAMBDA_2, GAMMA_2, PHI_2 = p_list[i+1]
    k_LAMBDA_Diff =  k * (LAMBDA_1 - LAMBDA_2)
    k_GAMMA_Diff =  k * (GAMMA_1 - GAMMA_2)
    k_PHI_Diff = [k*(curr - next) for (curr,next) in zip(PHI_1,PHI_2)]

    return k_LAMBDA_Diff, k_GAMMA_Diff, k_PHI_Diff
    
# compute tangent angle
def tau(i, p_list, train_articles, train_nonzero_idxs, C):
    #p_list is a list of lists of parameters in which a list is [LAMBDA, GAMMA, PHI]
    LAMBDA_1, GAMMA_1, PHI_1 = p_list[i]
    LAMBDA_0, GAMMA_0, PHI_0 = p_list[i-1]
    LAMBDA_2, GAMMA_2, PHI_2 = p_list[i+1]

    if compute_elbo( LAMBDA_2, GAMMA_2, PHI_2, train_articles, train_nonzero_idxs, C) > compute_elbo( LAMBDA_0, GAMMA_0, PHI_0 , train_articles, train_nonzero_idxs, C):
        flat_Lam = np.reshape((LAMBDA_2-LAMBDA_1).flatten(),((LAMBDA_2-LAMBDA_1).flatten().shape[0],1))
        #print(flat_Lam.shape)
        flat_Gam = np.reshape((GAMMA_2 - GAMMA_1).flatten(),((GAMMA_2 - GAMMA_1).flatten().shape[0],1))
        #print(flat_Gam.shape)
        flat_vector = np.vstack((flat_Lam, flat_Gam))
        #print(flat_vector.shape)

        #flat_Phi = (PHI_2-PHI_1).flatten()
        dif_PHI = [Phi_2-Phi_1 for (Phi_2,Phi_1) in zip(PHI_2,PHI_1)]
        for dif in dif_PHI:
            flat_dif = dif.flatten().reshape(dif.flatten().shape[0],1)
            #print(flat_dif.shape)
            flat_vector = np.vstack((flat_vector, flat_dif))

        norm = np.linalg.norm(flat_vector)+ 1e-30

        lam_dif = (LAMBDA_2-LAMBDA_1)/norm
        gam_dif = (GAMMA_2 - GAMMA_1)/norm
        PHI_dif = [dif_phi/norm for dif_phi in dif_PHI]

        return [lam_dif, gam_dif, PHI_dif]
    #TODO: what is the right way to normalize
    else:
        flat_Lam = np.reshape((LAMBDA_1-LAMBDA_0).flatten(),((LAMBDA_1-LAMBDA_0).flatten().shape[0],1))
        #print(flat_Lam.shape)
        flat_Gam = np.reshape((GAMMA_1 - GAMMA_0).flatten(),((GAMMA_1 - GAMMA_0).flatten().shape[0],1))
        #print(flat_Gam.shape)
        flat_vector = np.vstack((flat_Lam, flat_Gam))

        #flat_Phi = (PHI_1-PHI_0).flatten()
        dif_PHI = [Phi_1-Phi_0 for (Phi_1,Phi_0) in zip(PHI_1,PHI_0)]
        for dif in dif_PHI:
            flat_dif = dif.flatten().reshape(dif.flatten().shape[0],1)
            flat_vector = np.vstack((flat_vector, flat_dif))

        norm = np.linalg.norm(flat_vector)+ 1e-30
        lam_dif = (LAMBDA_1-LAMBDA_0)/norm
        gam_dif = (GAMMA_1 - GAMMA_0)/norm
        PHI_dif = [dif_phi/norm for dif_phi in dif_PHI]

        return [lam_dif, gam_dif, PHI_dif]
        #return [(LAMBDA_1-LAMBDA_0)/norm, (GAMMA_1 - GAMMA_0)/norm, [dif_phi for dif_phi in dif_PHI/norm]]
        
def main():

p_list = init(theta1, theta2, N)

for e in epochs:
    for p in p_list:
      LAMBDA, GAMMA, PHI = p
      compute_elbo(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C)

if __name__ == "__main__":
    main()
