from santaspkg.constants import *
from santaspkg.dataset import desired, family_size, penalties, N_FAMILIES

import numpy as np


class AssignmentHelper(object):
    def __init__(self, other=None):
        if other is None:
            self.non_copy_constructor()
        else:
            self.copy_constructor(other)

    def non_copy_constructor(self):
        # Index 0 of daily_occupancy is unused.
        self.daily_occupancy = np.zeros(N_DAYS+1, dtype=np.int16)
        self.assignment = np.full(N_FAMILIES, -1, dtype=np.int8)

        self.n_days_below_min = N_DAYS
        self.n_days_above_max = 0

    def copy_constructor(self, other):
        self.daily_occupancy  = np.copy(other.daily_occupancy)
        self.assignment       = np.copy(other.assignment)

        self.n_days_below_min = other.n_days_below_min
        self.n_days_above_max = other.n_days_above_max

    def assign(self, fam_id, day):
        assert self.assignment[fam_id] == -1, \
                'AssignmentHelper doesn\'t support reassignment of families.'
        assert 1 <= day <= N_DAYS, f'Must assign to a day in the range [1, {N_DAYS}].'
        self.assignment[fam_id] = day

        prev_occupancy = self.daily_occupancy[day]
        self.daily_occupancy[day] += family_size[fam_id]
        new_occupancy = self.daily_occupancy[day]

        if prev_occupancy < MIN_OCCUPANCY <= new_occupancy:
            self.n_days_below_min -= 1
        if prev_occupancy <= MAX_OCCUPANCY < new_occupancy:
            self.n_days_above_max += 1

    def copy(self):
        return AssignmentHelper(other=self)

    @staticmethod
    def accounting_helper(N_d, N_dp1):
        return max(0, (N_d - 125.0) / 400.0 * N_d**(0.5 + abs(N_d - N_dp1) / 50.0))

    def marginal_accounting_cost(self, fam_id, day):
        if day == 1:
            N_d   = self.daily_occupancy[day]
            N_dp1 = self.daily_occupancy[day + 1]

            current_cost = self.accounting_helper(N_d, N_dp1)

            N_d += family_size[fam_id]
            updated_cost = self.accounting_helper(N_d, N_dp1)
        elif day == N_DAYS:
            N_dm1 = self.daily_occupancy[day - 1]
            N_d   = self.daily_occupancy[day]
            N_dp1 = N_d

            current_cost = self.accounting_helper(N_dm1, N_d)
            current_cost += self.accounting_helper(N_d, N_dp1)

            N_d += family_size[fam_id]
            N_dp1 = N_d
            updated_cost = self.accounting_helper(N_dm1, N_d)
            updated_cost += self.accounting_helper(N_d, N_dp1)
        elif 1 < day < N_DAYS:
            N_dm1 = self.daily_occupancy[day - 1]
            N_d   = self.daily_occupancy[day]
            N_dp1 = self.daily_occupancy[day + 1]

            current_cost = self.accounting_helper(N_dm1, N_d)
            current_cost += self.accounting_helper(N_d, N_dp1)

            N_d += family_size[fam_id]
            updated_cost = self.accounting_helper(N_dm1, N_d)
            updated_cost += self.accounting_helper(N_d, N_dp1)
        else:
            raise ValueError(f'Must choose a day in the interval [1, {N_DAYS}].')

        return updated_cost - current_cost

    def marginal_accounting_costs(self, fam_id):
        '''
        Calculate the marginal accounting cost for this family at each day.
        Return result as a numpy arrray.
        '''
        raise NotImplementedError()

    def marginal_accounting_costss(self):
        '''
        Calculate the marginal accounting cost for any size family at each day.
        Returns a 2-D numpy array.
        '''
        raise NotImplementedError()

    def fits_below_max(self, fam_id, day):
        return (self.daily_occupancy[day] + family_size[fam_id] <= MAX_OCCUPANCY)


def greedy(family_order):
    ah = AssignmentHelper()

    family_iter = iter(family_order)

    # stage one
    while ah.n_days_below_min > 0:
        fam_id = next(family_iter)
        day_assigned = False
        for day in desired[fam_id]:
            if ah.daily_occupancy[day] < MIN_OCCUPANCY:
                ah.assign(fam_id, day)
                day_assigned = True
                break
        if not day_assigned:
            day = np.argmin(ah.daily_occupancy[1:]) + 1
            ah.assign(fam_id, day)

    # stage two
    for fam_id in family_iter:
        consideration_mask = np.array([ah.fits_below_max(fam_id, day) for day in desired[fam_id]])

        if not consideration_mask.any():
            consideration_mask = np.array([ah.fits_below_max(fam_id, day) for day in range(1, N_DAYS+1)])
            days_considered = np.array(list(range(1, N_DAYS+1)))[consideration_mask]
            marginal_accounting_costs = \
                    np.array([ah.marginal_accounting_cost(fam_id, day) for day in days_considered])
            day_idx = np.argmin(marginal_accounting_costs)
        else:
            days_considered = desired[fam_id, consideration_mask]
            marginal_accounting_costs = \
                    np.array([ah.marginal_accounting_cost(fam_id, day) for day in days_considered])
            marginal_consolation_costs = penalties[family_size[fam_id], :-1][consideration_mask]
            day_idx = np.argmin(marginal_accounting_costs + marginal_consolation_costs)

        day = days_considered[day_idx]
        ah.assign(fam_id, day)

    return ah.assignment


families_in_order = np.array(list(range(N_FAMILIES)), dtype=np.int16)


def random_greedy(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    family_order = random_state.permutation(families_in_order)
    assignment = greedy(family_order)
    return assignment
