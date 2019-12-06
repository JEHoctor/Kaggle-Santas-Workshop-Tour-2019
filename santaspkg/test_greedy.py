from santaspkg.greedy import random_greedy, AssignmentHelper, families_in_order, greedy_v2
from santaspkg.constants import *
from santaspkg.dataset import desired, family_size
from santaspkg.cost_function.tables import penalties
from santaspkg.cost_function.crescenzi import jited_cost
from santaspkg.cost_function import reference_cost_function
from santaspkg.dataset import sample_assignment
from santaspkg.simple_scheduler import initialize

import numpy as np


def test_random_greedy():
    rs = np.random.RandomState(500)
    best_score = 10**8
    for x in range(100):
        assignment = random_greedy(rs)
        score, n_out_of_range = jited_cost(assignment, desired, family_size, penalties)
        assert n_out_of_range == 0
        best_score = min(score, best_score)
    print('Best score during testing:', best_score)


def test_marginal_accounting_cost():
    # a very interesting number
    rs = np.random.RandomState(1729)
    family_order = rs.permutation(families_in_order)

    ah = AssignmentHelper()
    for fam_id in family_order[:-1]:
        ah.assign(fam_id, sample_assignment[fam_id])

    last_fam = family_order[-1]
    marginal_accounting_costs = \
            np.array([ah.marginal_accounting_cost(last_fam, day) for day in range(1, N_DAYS+1)])

    fam_penalties = penalties[family_size[last_fam]]
    marginal_consolation_costs = \
            np.full_like(marginal_accounting_costs, fill_value=fam_penalties[-1])
    for day, penalty in zip(desired[last_fam], fam_penalties):
        marginal_consolation_costs[day-1] = penalty

    marginal_costs = marginal_accounting_costs + marginal_consolation_costs

    def get_true_cost(day):
        new_ah = ah.copy()
        new_ah.assign(last_fam, day)
        return reference_cost_function(new_ah.assignment)

    costs = np.array([get_true_cost(day) for day in range(1, N_DAYS+1)])

    err_std = (costs - marginal_costs).std()
    assert err_std < 1e-6

def test_greedy_v2():
    assignment = greedy_v2(families_in_order)
    assignment2, _ = initialize()
    assignment2 = np.array(assignment2)
    assert np.array_equal(assignment, assignment2)
