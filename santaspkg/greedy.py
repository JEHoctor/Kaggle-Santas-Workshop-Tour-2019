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
        assert not self.is_assigned(fam_id), \
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

    def is_assigned(self, fam_id):
        return (self.assignment[fam_id] != -1)

    def is_done(self):
        return (self.assignment != -1).all()

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


def greedy_v2(family_order):
    # strongly inspired by https://www.kaggle.com/dan3dewey/santa-s-simple-scheduler
    ah = AssignmentHelper()

    # phase 1
    lower_days = [62,63,64,65, 69,70,71,72, 76,77,78,79, 83,84,85,86, 90,91,92,93, 97,98,99,100,
                  20,21,22,23, 27,28,29,30, 34,35,36,37, 41,42,43,44, 48,49,50,51, 55,56,57,58]

    max_ppl_day = 126+25

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4,5,4,3,6,7,6,7,4,5,6,8,7,4,8,5,6,7,8,6,7,8,8,9,9]
    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6,5,4,0,6,7,5,7,3,3,4,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for ich, nppl_min in zip(ichs, nppl_mins):
        for fam_id in family_order:
            day_ich = desired[fam_id, ich]
            nppl = family_size[fam_id]
            if ((not ah.is_assigned(fam_id)) and
                    (day_ich in lower_days) and
                    (nppl >= nppl_min) and
                    (ah.daily_occupancy[day_ich] < max_ppl_day)):
                ah.assign(fam_id, day_ich)

    # phase 2
    lower_days = [62,63,64,65, 69,70,71,72, 76,77,78,79, 83,84,85,86, 90,91,92,93, 97,98,99,100]

    max_ppl_day = 220
    max_ppl_above = 170

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4,5,4,3,6,7,6,7,4,5,6,8,7,4,8,5,6,7,8,6,7,8] #,8,9,9]
    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6,5,4,0,6,7,5,7,3,3,4,7,5,0,6,0,3,3,4,0,0,3] #,0,7,0]

    for ich, nppl_min in zip(ichs, nppl_mins):
        for fam_id in family_order:
            day_ich = desired[fam_id, ich]
            nppl = family_size[fam_id]
            if day_ich < 59:
                ppl_limit = max_ppl_day
            else:
                ppl_limit = max_ppl_above
            if ((not ah.is_assigned(fam_id)) and
                    (day_ich not in lower_days) and
                    (nppl >= nppl_min) and
                    (ah.daily_occupancy[day_ich] < ppl_limit)):
                ah.assign(fam_id, day_ich)

    # phase 3
    max_ppl_day = 260
    max_ppl_above = 210

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4,5,4,3,6,7,6,7,4,5,6,8,7,4,8,5,6,7,8,6,7,8,8,9,9]
    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6,5,4,0,6,7,5,7,3,3,4,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for ich, nppl_min in zip(ichs, nppl_mins):
        for fam_id in family_order:
            day_ich = desired[fam_id, ich]
            nppl = family_size[fam_id]
            if day_ich < 59:
                ppl_limit = max_ppl_day
            else:
                ppl_limit = max_ppl_above
            if ((not ah.is_assigned(fam_id)) and
                    (nppl >= nppl_min) and
                    (ah.daily_occupancy[day_ich] < ppl_limit)):
                ah.assign(fam_id, day_ich)
        if ah.is_done():
            break

    return ah.assignment


def random_greedy_v2(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    family_order = random_state.permutation(families_in_order)
    assignment = greedy_v2(family_order)
    return assignment
