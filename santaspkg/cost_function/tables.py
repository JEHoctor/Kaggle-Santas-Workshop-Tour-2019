from santaspkg.constants import *
from santaspkg.dataset import desired, family_size, N_FAMILIES


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
penalties.flags.writeable = False


choice_array = np.full((N_FAMILIES, N_DAYS + 1), -1)
for fam_id, fam_choices in enumerate(desired):
    for choice_number, day in enumerate(fam_choices):
        choice_array[fam_id, day] = choice_number
choice_array.flags.writeable = False


penalty_matrix = np.zeros_like(choice_array)
for fam_id in range(N_FAMILIES):
    choice_array_row = choice_array[fam_id]
    fam_size = family_size[fam_id]
    for day in range(N_DAYS + 1):
        penalty_matrix[fam_id, day] = penalties[fam_size, choice_array_row[day]]
penalty_matrix.flags.writeable = False


accounting_matrix = np.zeros((MAX_OCCUPANCY + 1, MAX_OCCUPANCY + 1))
for occupancy_count in range(1, MAX_OCCUPANCY + 1):
    for difference in range(MAX_OCCUPANCY + 1):
        accounting_cost = (occupancy_count - 125.0) / 400.0 * occupancy_count**(0.5 + difference / 50.0)
        accounting_matrix[occupancy_count, difference] = max(0, accounting_cost)
accounting_matrix.flags.writeable = False
