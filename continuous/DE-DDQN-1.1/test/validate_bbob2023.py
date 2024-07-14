from numpy.random import rand
import numpy as np
import math
import validate_dqn
import random
from cocopp import bbob
import cocoex
import time


class bbob2023problem():

    def __init__(self, f):
        self.objective_function = f
    
    def __str__(self):
        return self.objective_function.id 

    def __call__(self,x):
        return self.objective_function(x)

# suite = cocoex.Suite('bbob', "year:2023", "dimensions:10 instance_indices:1")
# print(len(suite),suite,suite.__module__)
# print(suite.__module__)
# for problem in suite:
#     print(problem, problem.dimension)
    # x0 = problem.initial_solution
    # print(problem(x0),problem.upper_bound,problem.lower_bounds)
    # 
    # 


# d = [10, 20]



st = time.time()
benchmark_name = 'bbob2023'


wu=0.5
wl=0.1
Alpha=0.01
Beta=0.01
P_max=0.85
use_joint = True # apply stateless on state-based or not

use_random = False # is stateless random or not

write_log = True
award_global_improvement = True
weight_update_rule = 1
seeds = range(30)
d = [20]
# d = [10]
folder = './left/'
for i in range(len(d)):
    info = "dimensions:"+str(d[i])+" instance_indices:1"
    func_select = cocoex.Suite('bbob', "year:2023", info)
    print('------- ', benchmark_name,len(func_select),' functions ------')
    # for j in range(16,len(func_select)):
    # for j in range(13,16):
    for j in range(19,len(func_select)):
        dim = d[i]
        fun = bbob2023problem(func_select[j])
        # fun = funcall
        print(fun)
        ubounds = func_select[j].upper_bounds
        lbounds = func_select[j].lower_bounds
        # lbounds = [-100 for _ in range(dim)]
        # ubounds = [100 for _ in range(dim)]
        lbounds = np.array(lbounds)
        print(lbounds)
        ubounds = np.array(ubounds)
        print(ubounds)
        best_value = 0# not used
        # bbob_f020_i01_d10_10/log_joint_FIR_AP_0.5_0.1_0.01_0.01_0.85_online_joint
        #/bbob_f020_i01_d20_20/log_joint_FIR_AP_0.5_0.1_0.01_0.01_0.85_online_joint

        # validate_dqn.DE_online_updated(fun, lbounds, ubounds, dim, best_value,benchmark_name,random_seeds=seeds,folder=folder,eps=0.05,\
        #                                wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
        #                                 award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
        validate_dqn.DE(fun, lbounds, ubounds, dim, best_value,benchmark_name,check_overwrite_log=True,folder=folder,\
                                       wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
                                        award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
print("run time (hour): ", (time.time()-st)/3600)



# random.seed(1)
# np.random.seed(1)
# d = 10
# fun = f16.F16(d)
# lbounds = fun.min_bounds; lbounds = np.array(lbounds); print(lbounds)
# ubounds = fun.max_bounds; ubounds = np.array(ubounds); print(ubounds)
# opti = fun.get_optimal_solutions()
# for o in opti:
#     print(o.phenome, o.objective_values)
# sol = np.copy(o.phenome)
# best_value = fun.objective_function(sol)
# validate_dqn.DE(fun, lbounds, ubounds, d, best_value)