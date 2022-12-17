'''
LDA 

Reference: https://github.com/taekb/lda\_cavi\\ 
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

from Data_Loader import load_data


'''
hyper parameters
'''

# Topic Exch. Dirichlet hyperparameter (mixture components)
ETA = 0.01 # Set to 100/V

# Topic Proportion Dirichlet hyperparameter (mixture proportions)
ALPHA = 0.1 # Set to

# Maximum number of iterations (Might not use)
MAX_ITER = 100

# Number of CAVI iterations
N_TRIALS = 1

# Test data size
N_TEST = 10



def init_var_param(train_articles, C):
    print('Initializing variational parameters...')

    # Number of articles, vocabulary size
    N, V = train_articles.shape

    # Topics (initializing LAMBDA for BETA)
    LAMBDA = np.random.uniform(low=0.01, high=1.0, size=(C,V))

    # Topic Proportions (initializing GAMMA for THETA)
    GAMMA = np.ones((N,C)) # Uniform prior

    # Topic Assignments (initializing PHI for Z)
    # Shape: (N,n_words,C) (Note: n_words is vacriable)
    PHI = []

    for article in train_articles:
        n_words = np.sum((article > 0).astype('int32'))
        article_phi = np.ones((n_words,C))
        article_phi = article_phi / C # Initialize to 1/C

        PHI.append(article_phi)

    return LAMBDA, GAMMA, PHI

# Compute ELBO
def compute_elbo(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C):
    elbo = 0

    # Number of articles, vocabulary size
    N, V = train_articles.shape

    # Add expected log joint
    ## First term: \sum_{k=1}^C E[log p(BETA_k)]
    E_log_p_beta = 0
    for k in range(C):
        E_log_p_beta += (ETA-1) * np.sum(digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k])))

    elbo += E_log_p_beta

    ## Second term: \sum_{i=1}^N E[log p(THETA_i)]
    E_log_p_theta = 0
    for i in range(N):
        E_log_p_theta += (ALPHA-1) * np.sum(digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i])))

    elbo += E_log_p_theta

    ## Third term:
    ## \sum_{i=1}^N \sum_{j=1}^M \sum_{k=1}^C
    ## (E[log p(Z_ij|THETA_i)] + E[log p(X_ij)|BETA,Z_ij)])
    E_log_p_xz = 0
    for i in range(N):
        article = train_articles[i]
        nonzero_idx = train_nonzero_idxs[i]

        corr_idx = 0

        for idx in nonzero_idx:
            ### E[log p(Z_ij|THETA_i)]
            E_log_p_xz += article[idx] * np.sum(PHI[i][corr_idx] * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))

            ### E[log p(X_ij|BETA,Z_ij)]
            E_log_p_xz += article[idx] * np.sum(PHI[i][corr_idx] * (digamma(LAMBDA[:,idx]) - digamma(np.sum(LAMBDA, axis=1))))

            corr_idx += 1

        # Check if number of updates match with number of words
        assert(corr_idx == len(nonzero_idx))

    elbo += E_log_p_xz

    # Add entropy
    ## Fourth term: -\sum_{k=1}^C E[log q(BETA_k)]
    E_log_q_beta = 0
    for k in range(C):
        E_log_q_beta += -loggamma(np.sum(LAMBDA[k])) + np.sum(loggamma(LAMBDA[k]))
        E_log_q_beta += -np.sum((LAMBDA[k]-1) * (digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k]))))

    elbo += E_log_q_beta

    ## Fifth term: -\sum_{i=1}^N E[log q(THETA_i)]
    E_log_q_theta = 0
    for i in range(N):
        E_log_q_theta += -loggamma(np.sum(GAMMA[i])) + np.sum(loggamma(GAMMA[i]))
        E_log_q_theta += -np.sum((GAMMA[i]-1) * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))

    elbo += E_log_q_theta

    ## Sixth term: -\sum_{i=1}^N \sum_{j=1}^M (E[log q(Z_ij)])
    E_log_q_z = 0
    for i in range(N):
        article = train_articles[i]
        nonzero_idx = train_nonzero_idxs[i]

        corr_idx = 0
        for idx in nonzero_idx:
            #print(PHI[i][corr_idx])
            #print(type(PHI[i][corr_idx]))
            #print(PHI[i][corr_idx]+1e-1)
            E_log_q_z += -article[idx] * np.sum(PHI[i][corr_idx] * np.log((PHI[i][corr_idx]+1e-2)/(np.linalg.norm(PHI[i][corr_idx]+1e-2)+1e-20)))
            #E_log_q_z += -article[idx] * np.sum(PHI[i][corr_idx] * np.log(np.minimum(PHI[i][corr_idx],1e-9)))/(np.linalg.norm(PHI[i][corr_idx])+1e-20)
            #E_log_q_z += -article[idx] * np.sum(PHI[i][corr_idx] * np.log(np.abs(PHI[i][corr_idx])))

            corr_idx += 1

        # Check if number of updates match with number of words
        assert(corr_idx == len(nonzero_idx))

    elbo += E_log_q_z

    print('ELBO: {}'.format(elbo))

    return elbo

# Runs CAVI for LDA
# TODO: Change max iteration to convergence criterion
def run_cavi(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C, max_iter, predict_flag=False):
    # Unpack initial variational parameters
    LAMBDA_t = copy.deepcopy(LAMBDA) # Shape: (C,V)
    GAMMA_t = copy.deepcopy(GAMMA) # Shape: (N,C)
    PHI_t = copy.deepcopy(PHI) # Shape: (N,n_words,C)

    # Number of articles, vocabulary size
    N, V = train_articles.shape

    elbos = []

    print('Running CAVI for LDA (C: {}, Iter: {})...'.format(C, max_iter))
    for t in range(max_iter):
        print('Iteration {}'.format(t+1))
        print('Updating PHI and GAMMA')

        # For each document
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

        # Compute ELBO
        elbo = compute_elbo(LAMBDA_t, GAMMA_t, PHI_t, train_articles, train_nonzero_idxs, C)
        elbos.append(elbo)

    LAMBDA_final = copy.deepcopy(LAMBDA_t)
    GAMMA_final = copy.deepcopy(GAMMA_t)
    PHI_final = copy.deepcopy(PHI_t)

    return LAMBDA_final, GAMMA_final, PHI_final, elbos

def filter_train_words(idx_to_words, train_nonzero_idxs):
    # Number of train articles, vocabulary size
    N, V = train_articles.shape

    train_word_idxs = []

    for i in range(N):

        nonzero_idx = train_nonzero_idxs[i]
        train_word_idxs += nonzero_idx

    train_word_idxs = set(train_word_idxs)
    test_word_idxs = set(range(V)) - train_word_idxs

    return test_word_idxs

# Computes the predictive likelihood score on held out articles
def compute_pred_score(LAMBDA, GAMMA, PHI, test_articles, test_nonzero_idxs, test_word_idxs):
    # Number of test articles, vocabulary size
    N,V = test_articles.shape

    C = LAMBDA.shape[0]

    # Expected BETA
    BETA = np.zeros((C,V))

    for k in range(C):
        BETA[k] = LAMBDA[k] / np.sum(LAMBDA[k])

    # Expected THETA
    THETA = np.zeros((N,C))

    for i in range(N):
        THETA[i] = GAMMA[i] / np.sum(GAMMA[i])

    # Compute predictive likelihood
    print('Computing predictive likelihood score on held out documents...')

    score = 0
    for i in tqdm(range(N)):
        article = test_articles[i]
        nonzero_idx = test_nonzero_idxs[i]

        for idx in nonzero_idx:
            if not idx in test_word_idxs:
                score += np.log(np.dot(THETA[i], BETA[:,idx]))

    print('Predictive likelihood: {}'.format(score))

    return score


def main():
C = 10 #num of topics
#use_cached = kwargs['use_cached']
predict_flag = True
max_iter = 20
n_trials = 1
trim = 0.01

start_time = datetime.now()

global ALPHA
ALPHA = 1 / C

# Load data
print('Loading AP article data...')
idx_to_words, articles, nonzero_idxs = load_data(BOW_PATH,VOCAB_PATH, trim)

# Split data into train/test

train_articles = articles[:-N_TEST]
train_nonzero_idxs = nonzero_idxs[:-N_TEST]
test_articles = articles[-N_TEST:]
test_nonzero_idxs = nonzero_idxs[-N_TEST:]

#var_param_dict = {}
colors = matplotlib.cm.tab10(np.linspace(0,1,n_trials))
best_trial = 0
best_elbo = -np.inf

for trial in range(n_trials):
    print('Running LDA CAVI {}/{}...'.format(trial+1, n_trials))

    # Initialize variational parameters
    LAMBDA, GAMMA, PHI = init_var_param(train_articles, C)

    # Run CAVI
    LAMBDA_final, GAMMA_final, PHI_final, elbos = run_cavi(LAMBDA, GAMMA, PHI, train_articles,
                                                            train_nonzero_idxs, C, max_iter)

    elbo_final = elbos[-1]

    if elbo_final > best_elbo:
        best_trial = trial
        best_elbo = elbo_final
        
    '''
    var_param_dict[trial] = {'LAMBDA': LAMBDA_final,
                              'GAMMA': GAMMA_final,
                              'PHI': PHI_final,
                              'elbos': elbos}
    '''
    # Plot ELBO
    plt.plot(np.arange(0, max_iter), elbos, label=trial+1, color=colors[trial])

#plt.legend()
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.show()
plt.savefig('./elbo_plot_{}_{}.png'.format(C, max_iter))
print('ELBO plot saved.')

# Compute predictive likelihood score on held out test articles
if predict_flag:
    
    #LAMBDA_fixed = var_param_dict[best_trial]['LAMBDA']
    LAMBDA_fixed = LAMBDA_final

    # Initialize variational parameters
    _, GAMMA, PHI = init_var_param(test_articles, C)

    # Run CAVI
    LAMBDA_pred, GAMMA_pred, PHI_pred, elbos = run_cavi(LAMBDA_fixed, GAMMA, PHI, test_articles,
                                                        test_nonzero_idxs, C, max_iter, predict_flag=True)
    test_word_idxs = []
    # Compute predictive likelihood
    test_word_idxs = filter_train_words(idx_to_words, train_nonzero_idxs)
    pred_score = compute_pred_score(LAMBDA_pred, GAMMA_pred, PHI_pred, test_articles,
                                    test_nonzero_idxs, test_word_idxs)
    #var_param_dict['score'] = pred_score
    pred_score_list.append(pred_score)

# Save inferred variational parameters
'''
with open('var_params_{}_{}.pkl'.format(C, max_iter), 'wb') as fh:
    pickle.dump(var_param_dict, fh)
    print('Inferred variational parameters saved.')
'''

end_time = datetime.now()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time.seconds, 3600)
minutes, seconds = divmod(rem, 60)
print('Total run-time: {}:{}:{}'.format(hours, minutes, seconds))


if __name__ == "__main__":
    main()
