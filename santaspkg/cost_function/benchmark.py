# This module is for determining which cost function implementation is the fastest. Correctness
# of each implementation is checked in a test_*.py file.
# run with `python -m santaspkg.cost_function.benchmark`

import timeit
import numpy as np
import pandas as pd
from santaspkg import dataset

from santaspkg.cost_function.reference_cost_function import reference_cost_function
from santaspkg.cost_function.four_x_faster import cost_function as four_x_faster_cost_function
from santaspkg.cost_function.crescenzi_cost_function import cost_function as crescenzi_cost_function

from collections import namedtuple


FunctionDetails = namedtuple(
    'FunctionDetails',
    ['name', 'f', 'input_type', 'warm_f'],
    defaults=(False,),
)


named_functions = [
    FunctionDetails('reference', reference_cost_function, 'list'),
    FunctionDetails('4x faster', four_x_faster_cost_function, 'list'),
    FunctionDetails('crescenzi', crescenzi_cost_function, 'numpy', warm_f=True),
]


def main():
    submission = dataset.sample_submission
    prediction_as_list = submission['assigned_day'].tolist()
    prediction_as_numpy = np.asarray(prediction_as_list)

    for function_name, f, input_type, warm_f in named_functions:
        if input_type == 'list':
            prediction_correct_type = prediction_as_list
        elif input_type == 'numpy':
            prediction_correct_type = prediction_as_numpy
        else:
            raise ValueError('For each named function, the input_type must be \'list\' or \'numpy\'.')

        if warm_f:
            f(prediction_correct_type)

        times = timeit.repeat(
            'f(prediction)',
            number=10**2,
            globals={
                'f': f,
                'prediction': prediction_correct_type
            })
        print(function_name, np.mean(times), np.std(times))


if __name__ == '__main__':
    main()
