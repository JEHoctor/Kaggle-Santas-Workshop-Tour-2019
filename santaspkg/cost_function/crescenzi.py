# The author of this cost function implementation claims it is 250x faster.
# https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit

from santaspkg.dataset import desired, family_size
from santaspkg.cost_function.tables import penalties

import numpy as np
import pandas as pd
from numba import njit


def cost_function(assignment):
    penalty, n_out_of_range = jited_cost(np.asarray(assignment), desired, family_size, penalties)
    assert n_out_of_range == 0
    return penalty


def soft_cost_function(assignment):
    penalty, n_out_of_range = jited_cost(np.asarray(assignment), desired, family_size, penalties)
    return penalty + (10**8) * n_out_of_range


@njit()
def jited_cost(assignment, desired, family_size, penalties):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    # dtype=np.int16 could work and save hella space
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i in range(len(assignment)):
        n = family_size[i]
        pred = assignment[i]
        n_choice = 0
        for j in range(len(desired[i])):
            if desired[i, j] == pred:
                break
            else:
                n_choice += 1

        daily_occupancy[pred - 1] += n
        penalty += penalties[n, n_choice]

    accounting_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

    penalty += accounting_cost
    return np.asarray([penalty, n_out_of_range])
