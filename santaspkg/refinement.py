from santaspkg.cost_function import soft_cost_function
from santaspkg.dataset import desired
import numpy as np


def refinement_pass(prediction):
    best = np.array(prediction, copy=True)
    best_score = soft_cost_function(best)

    # loop over each family
    for fam_id in range(len(best)):
        # loop over each family choice
        for day in desired[fam_id]:
            new = np.copy(best)
            new[fam_id] = day # add in the new pick
            new_score = soft_cost_function(new)
            if new_score < best_score:
                best = new
                best_score = new_score

    return best


def refine_until_convergence(prediction):
    new_prediction = refinement_pass(prediction)
    while not np.array_equal(new_prediction, prediction):
        prediction = new_prediction
        new_prediction = refinement_pass(prediction)
    return new_prediction


def refine_n(prediction, N):
    ''' refine_n; refine until convergence with at most N passes'''
    new_prediction = refinement_pass(prediction)
    i = 1
    while not np.array_equal(new_prediction, prediction) and i < N:
        prediction = new_prediction
        new_prediction = refinement_pass(prediction)
        i += 1
    return new_prediction