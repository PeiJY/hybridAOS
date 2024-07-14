
# from cec2005real.cec2005 import Function
from numpy.random import rand
import numpy as np
import math
# from optproblems import cec2005
from optproblems.cec2005 import *
from optproblems import mpm
import cec2017.functions as functions
# import validate_dqn
import validate_dq_w_log as validate_dqn
import random
from optproblems import zdt
from optproblems import continuous
import time
from cocopp import bbob

#function no. = 1 to 25
# dim = 2, 10, 30, 50

# Total problem instances = 100

#def EA_AOS(fun, lbounds, ubounds, budget, problem_index):
#print("DE_AOS..",fun)
#cost =
#print("cost: ", cost)
#return cost

# FF = 0.5; CR = 1.0

# d = [2, 10, 30, 50]

# reward_cec1 = np.inf



class cec2017problem():

    def __init__(self, f):
        self.objective_function = f
    
    def __str__(self):
        return self.objective_function.__name__ 

    def __call__(self,x):
        return self.objective_function([x])[0]


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
folder  = './weight_log/'
# d = [10, 30, 50]
# func_select = functions.all_functions
func_select = [functions.f1]
# d = [50]
# d = [30]
d = [10]
st = time.time()
benchmark_name = 'cec2017'
seeds = range(30)
for i in range(len(d)):
    for j in range(len(func_select)):
    # for j in range(15,20):  
    # for j in range(5,10):
        dim = d[i]
        fun = cec2017problem(func_select[j])
        # fun = funcall
        print(fun)
        lbounds = [-100 for _ in range(dim)]
        ubounds = [100 for _ in range(dim)]
        lbounds = np.array(lbounds)
        print(lbounds)
        ubounds = np.array(ubounds)
        print(ubounds)
        best_value = 0# not used

        validate_dqn.DE_online_updated(fun, lbounds, ubounds, dim, best_value,benchmark_name,random_seeds=seeds,eps=0.05,folder = folder,\
                                       wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
                                        award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
        # validate_dqn.DE(fun, lbounds, ubounds, dim, best_value,benchmark_name,\
        #                                wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
        #                                 award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
        # validate_dqn.DE(fun, lbounds, ubounds, dim, best_value,benchmark_name)

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