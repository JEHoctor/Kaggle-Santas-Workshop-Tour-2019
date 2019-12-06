# The author of this cost function implementation claims it is 600x faster.
# https://www.kaggle.com/xhlulu/santa-s-2019-600x-faster-cost-function-23-s

from santaspkg.constants import *
from santaspkg.dataset import family_size
from santaspkg.cost_function.tables import penalty_matrix, accounting_matrix
from santaspkg.cost_function.crescenzi import soft_cost_function as crescenzi_soft_cost_function

import numpy as np
from numba import njit


days_array = np.arange(N_DAYS, 0, -1)
days_array.flags.writeable = False


def cost_function(assignment):
    '''
    This function ideally takes a numpy array of length N_FAMILIES. Each entry is
    a day number from 1 to 100. If a list is given instead, it will convert it to a
    numpy array.
    
    The function fails an assert on an invalid assignment of families to days, and
    returns the cost of the assignment otherwise.
    '''
    assignment = np.asarray(assignment)
    cost = jited_cost_function(assignment, family_size, days_array, penalty_matrix, accounting_matrix)
    assert cost != -1.0
    return cost


def soft_cost_function(assignment):
    '''
    This function ideally takes a numpy array of length N_FAMILIES. Each entry is
    a day number from 1 to 100. If a list is given instead, it will convert it to a
    numpy array.
    
    If the assignment of families to days is valid, it returns the cost, otherwise
    it produces a large value that penalizes the assignment in proportion to how
    many invalid days it has.
    '''
    assignment = np.asarray(assignment)
    cost = jited_cost_function(assignment, family_size, days_array, penalty_matrix, accounting_matrix)
    if cost == -1.0:
        return crescenzi_soft_cost_function(assignment)
    else:
        return cost


@njit
def jited_cost_function(assignment, family_size, days_array, penalty_matrix, accounting_matrix):
    N_FAMILIES = len(family_size)
    N_DAYS = 100
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    # We'll use this to count the number of people scheduled each day
    # Consider using np.uint16
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    penalty = 0
    
    # Looping over each family; d is the day, n is size of that family
    for i in range(N_FAMILIES):
        n = family_size[i]
        d = assignment[i]
        
        daily_occupancy[d] += n
        penalty += penalty_matrix[i, d]

    # for each date, check total occupancy 
    # Day 0 does not exist, so we do not count it
    relevant_occupancy = daily_occupancy[1:]
    incorrect_occupancy = np.any(
        (relevant_occupancy > MAX_OCCUPANCY) | 
        (relevant_occupancy < MIN_OCCUPANCY)
    )

    # This is the best we can do. The rest of this function can't handle
    # occupancies out of range so we have to quit now.
    if incorrect_occupancy:
        return -1.0

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    init_occupancy = daily_occupancy[days_array[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days_array, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days_array[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += accounting_matrix[today_count, diff]
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty
