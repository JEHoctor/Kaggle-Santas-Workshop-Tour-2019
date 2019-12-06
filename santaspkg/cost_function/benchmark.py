# This module is for determining which cost function implementation is the fastest. Correctness
# of each implementation is checked in a test_*.py file.
# run with `python -m santaspkg.cost_function.benchmark`

from santaspkg.cost_function.reference import reference_cost_function
from santaspkg.cost_function.crescenzi import cost_function as crescenzi_cost_function
from santaspkg.cost_function.xhlulu import cost_function as xhlulu_cost_function
from santaspkg.dataset import sample_assignment

import timeit
import numpy as np
import pandas as pd

from collections import namedtuple


FunctionDetails = namedtuple(
    'FunctionDetails',
    ['name', 'f', 'input_type', 'warm_f'],
    defaults=(False,),
)


named_functions = [
    FunctionDetails('reference', reference_cost_function, 'list'),
    FunctionDetails('crescenzi', crescenzi_cost_function, 'numpy', warm_f=True),
    FunctionDetails('xhlulu', xhlulu_cost_function, 'numpy', warm_f=True),
]


def main():
    assignment_as_list = list(sample_assignment)
    assignment_as_numpy = sample_assignment

    for function_name, f, input_type, warm_f in named_functions:
        if input_type == 'list':
            assignment_correct_type = assignment_as_list
        elif input_type == 'numpy':
            assignment_correct_type = assignment_as_numpy
        else:
            raise ValueError('For each named function, the input_type must be \'list\' or \'numpy\'.')

        if warm_f:
            f(assignment_correct_type)

        times = timeit.repeat(
            'f(assignment)',
            number=10**2,
            globals={
                'f': f,
                'assignment': assignment_correct_type
            })
        print(function_name, np.mean(times), np.std(times))


if __name__ == '__main__':
    main()
