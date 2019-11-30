import numpy as np
import pandas as pd
from santaspkg.constants import *
from santaspkg.cost_function import cost_function
from santaspkg.cost_function import soft_cost_function as cost_function
from santaspkg.dataset import data, sample_submission
from santaspkg.refinement import refine_until_convergence, refinement_pass

#Make objects to reference later

family_size_dict = data[['n_people']].to_dict()['n_people']

cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].to_dict()

# from 100 to 1
days = list(range(N_DAYS,0,-1))

# Start with the sample submission values
best = sample_submission['assigned_day'].tolist()

# Refine the sample submission
new = refinement_pass(best)

sample_submission['assigned_day'] = new
score = cost_function(new)
sample_submission.to_csv(f'./santa-workshop-tour-2019/submission_{score}.csv')
print(f'Score: {score}')
