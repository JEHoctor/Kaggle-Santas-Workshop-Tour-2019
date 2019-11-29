# The author of this cost function implementation claims it is 250x faster.
# https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit

import numpy as np
import pandas as pd 
from numba import njit
from santaspkg import dataset


data = pd.read_csv(dataset.family_data_file, index_col='family_id')


desired = data.values[:, :-1]
family_size = data.n_people.values
penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in range(family_size.max() + 1)
])


def cost_function(prediction):
    penalty, n_out_of_range = jited_cost(np.asarray(prediction), desired, family_size, penalties)
    return penalty


@njit()
def jited_cost(prediction, desired, family_size, penalties):
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i in range(len(prediction)):
        n = family_size[i]
        pred = prediction[i]
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
