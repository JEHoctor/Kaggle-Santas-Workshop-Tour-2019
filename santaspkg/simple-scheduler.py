import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Some constants
data_dir = "../input/santa-workshop-tour-2019/"
family_file = "family_data.csv"

NDAYS = 100
NFAMS = 5000
MAX_PPL = 300
MIN_PPL = 125

# The family preference cost parameters
PENALTY_CONST = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
PENALTY_PPL = [0, 0, 9, 9, 9, 18, 18, 36, 36, 199+36, 398+36]

# The seed is set once here at beginning of notebook.
RANDOM_SEED = 127
np.random.seed(RANDOM_SEED)

def family_cost(ichoice, nppl):
    global PENALTY_CONST, PENALTY_PPL
    return PENALTY_CONST[ichoice] + nppl*PENALTY_PPL[ichoice]

# Show the cost-per-person in matrix form
# Note that higher choice values can give lower per-person cost.
# Also created a dictionary mapping the choice-nppl tuple to cost_pp.
cost_pp_dict = {}
print("    Cost per Person")
print("\n       nppl= 2       3       4       5       6       7       8\nichoice")
# choices are 0 to 10
for ichoice in range(11):
    # numbers of people in a family are 2 to 8:
    choice_str = str(ichoice).rjust(5)+":"
    for nppl in range(2,9):
        cost_pp = family_cost(ichoice, nppl)/nppl
        cost_pp_dict[(ichoice,nppl)] = cost_pp
        choice_str = choice_str + str(int(cost_pp)).rjust(8)
    print(choice_str)

# Can use the cost_pp_dict to go through the ichoice, nppl combinations,
# in order from least to greatest cost-per-person, if that's useful.
# (Didn't use this, put values in by hand below.)
if False:
    sorted_cost_pp = sorted(cost_pp_dict.items(), key = 
             lambda kv:(kv[1], kv[0]))
    for ich_nppl in sorted_cost_pp:
        ichoice = ich_nppl[0][0]
        nppl = ich_nppl[0][1]
        print(ichoice,nppl)

# Define the accounting cost function
def accounting_cost(people_count):
    # people_count[iday] is an array of the number of people each day,
    # valid for iday=1 to NDAYS (iday=0 not used).
    total_cost = 0.0
    ppl_yester = people_count[NDAYS]
    for iday in range(NDAYS,0,-1):
        ppl_today = people_count[iday]
        ppl_delta = np.abs(ppl_today - ppl_yester)
        day_cost = (ppl_today - 125)*(ppl_today**(0.5+ppl_delta/50.0))/400.0
        total_cost += day_cost
        ##print("Day {}: delta = {}, $ {}".format(iday, ppl_delta, int(day_cost)))
        # save for tomorrow
        ppl_yester = people_count[iday]
    print("Total accounting cost: {:.2f}.  Ave costs:  {:.2f}/day,  {:.2f}/family".format(
        total_cost,total_cost/NDAYS,total_cost/NFAMS))
    return total_cost

# Read in the data
df_family = pd.read_csv(data_dir+family_file)
# The "choice_" column headings use a lot of room, change to "ch_"
the_columns = df_family.columns.values
for ich in range(10):
    the_columns[ich+1] = "ch_"+str(ich)
df_family.columns = the_columns

# Total number of people
total_people  = df_family['n_people'].sum()
# and average per day:
ave_ppl_day = int(total_people / NDAYS)
print("Total number of people visiting is {}, about {} per day".format(total_people, ave_ppl_day))

# Add an assigned day column, inititalize it to -1 (not assigned)
df_family['assigned_day'] = -1

# As the results of v1-v3 showed, there are certain days that are less subscribed than others.

# (v4) Fill using lowest to higher cost-per-person choices.
# Also fill the lower-demand days above day 60 first...
if True:
    sched_method = 'LowHighCpp'
    
    # Reset the assignements and the people_count_m1 array:
    df_family['assigned_day'] = -1
    people_count_m1 = np.zeros(NDAYS)
    
    print("\nFilling the low-request days above day 60 ...\n")
    # First, assign the lower-requested days.
    # The low-people days are every 4 out of 7.
    # The 6 low regions above day 60 are:
    lower_days = [62,63,64,65, 69,70,71,72, 76,77,78,79, 83,84,85,86, 90,91,92,93, 97,98,99,100]
    # include the 5 other low regions:
    lower_days = lower_days + [20,21,22,23, 27,28,29,30, 34,35,36,37, 41,42,43,44, 48,49,50,51, 55,56,57,58]
    # will fill these to the minimum needed, or a bit more:
    
    max_ppl_day = 126+25
    
    # Set the desired cost-per-person limit by specifying:
    # i) specific choice to use, and ii) a minimum number of people (inclusive)
    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4,5,4,3,6,7,6,7,4,5,6,8,7,4,8,5,6,7,8,6,7,8,8,9,9]
    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6,5,4,0,6,7,5,7,3,3,4,7,5,0,6,0,3,3,4,0,0,3,0,7,0]
    for icost in range(len(ichs)):
        ich = ichs[icost]
        ich_str = 'ch_'+str(ich)
        nppl_min = nppl_mins[icost]
        print("Doing ",ich_str,"  nppl >=",nppl_min)
        #
        # Go though the families and assign ones that meet the criteria
        for ifam in df_family.index:
            day_ich = df_family.loc[ifam,ich_str]
            nppl = df_family.loc[ifam,'n_people']
            if ((df_family.loc[ifam,'assigned_day'] < 0) and
                    (day_ich in lower_days) and (nppl >= nppl_min) and
                    (people_count_m1[day_ich-1] < max_ppl_day)):
                ##print(ifam,day_ich,nppl,sum(people_count_m1))
                # OK, got one. Assign it:
                df_family.loc[ifam,'assigned_day'] = day_ich
                # and keep track of the people count:
                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']
        print("\nTotal assigned families = ",sum(df_family['assigned_day'] > 0),
             " and people =",sum(people_count_m1))
    
    print("\nFilling all the rest of the days ...\n")
    # will fill the other days to a maximum amount, with a break above
    
    max_ppl_day = 220
    max_ppl_above = 170
    
    lower_days = [62,63,64,65, 69,70,71,72, 76,77,78,79, 83,84,85,86, 90,91,92,93, 97,98,99,100]
    # Set the desired cost-per-person limit by specifying:
    # i) specific choice to use, and ii) a minimum number of people (inclusive)
    # These look like enough to get 125 in each of the low
    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4,5,4,3,6,7,6,7,4,5,6,8,7,4,8,5,6,7,8,6,7,8] #,8,9,9]
    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6,5,4,0,6,7,5,7,3,3,4,7,5,0,6,0,3,3,4,0,0,3] #,0,7,0]
    for icost in range(len(ichs)):
        ich = ichs[icost]
        ich_str = 'ch_'+str(ich)
        nppl_min = nppl_mins[icost]
        print("Doing ",ich_str,"  nppl >=",nppl_min)
        #
        # Go though the families and assign ones that meet the criteria
        for ifam in df_family.index:
            day_ich = df_family.loc[ifam,ich_str]
            nppl = df_family.loc[ifam,'n_people']
            if day_ich < 59:
                ppl_limit = max_ppl_day
            else:
                ppl_limit = max_ppl_above
            if ((df_family.loc[ifam,'assigned_day'] < 0) and
                    not(day_ich in lower_days) and (nppl >= nppl_min) and
                    (people_count_m1[day_ich-1] < ppl_limit)):
                ##print(ifam,day_ich,nppl,sum(people_count_m1))
                # OK, got one. Assign it:
                df_family.loc[ifam,'assigned_day'] = day_ich
                # and keep track of the people count:
                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']
        print("\nTotal assigned families = ",sum(df_family['assigned_day'] > 0),
             " and people =",sum(people_count_m1))

    # Finally, the remaining families don't have any of their choices still available,
    # increase the people limits to get them in
    print("\nPut these last few anywhere ...\n")
    
    max_ppl_day = 260
    max_ppl_above = 210
    
    # Set the desired cost-per-person limit by specifying:
    # i) specific choice to use, and ii) a minimum number of people (inclusive)
    # These look like enough to get 125 in each of the low
    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4,5,4,3,6,7,6,7,4,5,6,8,7,4,8,5,6,7,8,6,7,8,8,9,9]
    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6,5,4,0,6,7,5,7,3,3,4,7,5,0,6,0,3,3,4,0,0,3,0,7,0]
    for icost in range(len(ichs)):
        ich = ichs[icost]
        ich_str = 'ch_'+str(ich)
        nppl_min = nppl_mins[icost]
        print("Doing ",ich_str,"  nppl >=",nppl_min)
        #
        # Go though the families and assign ones that meet the criteria
        for ifam in df_family.index:
            day_ich = df_family.loc[ifam,ich_str]
            nppl = df_family.loc[ifam,'n_people']
            if day_ich < 59:
                ppl_limit = max_ppl_day
            else:
                ppl_limit = max_ppl_above
            if ((df_family.loc[ifam,'assigned_day'] < 0) and
                    (nppl >= nppl_min) and
                    (people_count_m1[day_ich-1] < ppl_limit)):
                ##print(ifam,day_ich,nppl,sum(people_count_m1))
                # OK, got one. Assign it:
                df_family.loc[ifam,'assigned_day'] = day_ich
                # and keep track of the people count:
                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']
        print("\nTotal assigned families = ",sum(df_family['assigned_day'] > 0),
             " and people =",sum(people_count_m1))
        # Done?
        if (sum(df_family['assigned_day'] > 0) >= 5000):
            break

# Check for any not-assigned families
if df_family['assigned_day'].min() < 0:
    print("Ooops!  Some families did not get days assigned!")
    print("Number assigned = {}".format(sum(df_family['assigned_day'] > 0)))
    halt_on_this_routine()

# Write out the submission file:
df_family[['family_id','assigned_day']].to_csv("submission.csv", index=False)
