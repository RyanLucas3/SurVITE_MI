import numpy as np
from sklearn.linear_model import LogisticRegression

def at_risk_probability(tr_x, tr_t, tr_a):

    '''
    :param tr_x: patient covariates.
    :param tr_t: the time of event realization.
    :param tr_a: treatment that the patient actually took.
    :return: the at-risk probability.
    '''

    means = {} # Finding the average number of individuals at-risk beyond time tau

    # for {1,..., t_max}
    for τ in range(1, int(max(tr_t))):
        
        # find I[t_i ⋝ τ] for all patients i ∈ [n].
        tau_geq = is_greater_than_tau(tr_t, τ) 

        # find I[t_i ⋝ τ] × I[a_i = 1] for all patients.
        tau_a = np.array([tr_a[i]*tau_geq[i] for i in range(tr_x.shape[0])]) 

        # Collect average at-risk probability at time τ
        means[τ] = np.mean(tau_a)

    means = list(means.values())
    means = [means[0]] + list(means)

    return means

def propensity_score(tr_x, tr_a):

    '''
    :param tr_x: patient covariates.
    :param tr_a: treatment that the patient actually took.
    :return: the propensity score (the probability that a patient is in the treated group).
    '''

    # Logistic regression predicting P(A = a|X = x),
    clf = LogisticRegression('l2', class_weight='balanced',C=3.0)

    # Fit and output logistic probabilities.
    clf.fit(tr_x, tr_a)
    prob_all = clf.predict_proba(tr_x)

    return prob_all

def is_greater_than_tau(tr_t, tau):
    # {I[t_i ⋝ τ]: t ∈ tr_t}
    return np.array([1 if t >= tau else 0 for t in tr_t])
