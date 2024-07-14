
# from cec2005real.cec2005 import Function
from numpy.random import rand
import numpy as np
import math
# from optproblems import cec2005
from optproblems.cec2005 import *
from optproblems import mpm
import validate_dqn
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



# d = [10, 30]

# func_select = [unimodal.F3, basic_multimodal.F9, f16.F16, f18.F18, f23.F23] # test instance of DE-DDQN
# func_select = [unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8, basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13, expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24] # train instance of DE-DDQN
func_select = [unimodal.F3, basic_multimodal.F9, f16.F16, f18.F18, f23.F23, unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8, basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13, expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24] # all instances 
benchmark_name = 'cec2005'
st = time.time()
# func_select = [zdt.ZDT1, zdt.ZDT2,zdt.ZDT3, zdt.ZDT4,zdt.ZDT5, zdt.ZDT6]
# func_select = [continuous.Hartman6]
# func_select = [mpm.TestProblem]

func_select = [f21.F21, f22.F22, f24.F24]
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
# seeds = range(1,30,2)
seeds = range(30)
d = [30]
print(len(func_select))
folder = './left/'
for i in range(len(d)):
    for j in range(len(func_select)):
    # for j in [19,20]:
        dim = d[i]
        # fun = func_select[j](dim)
        fun = func_select[j](dim)
        print(fun)
        lbounds = fun.min_bounds; lbounds = np.array(lbounds); print(lbounds)
        ubounds = fun.max_bounds; ubounds = np.array(ubounds); print(ubounds)
        opti = fun.get_optimal_solutions()
        for o in opti:
            print('------ ',o.phenome, o.objective_values,' ------')
        sol = np.copy(o.phenome)
        best_value = fun.objective_function(sol)
        #print(" best value= ",best_value)
        #for repeat in range(10):
        validate_dqn.DE_online_updated(fun, lbounds, ubounds, dim, best_value,benchmark_name,random_seeds=seeds,folder=folder,\
                                       wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
                                        award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
        # validate_dqn.DE(fun, lbounds, ubounds, dim, best_value,benchmark_name,\
        #                                wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
        #                                 award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
        # validate_dqn.DE(fun, lbounds, ubounds, dim, best_value, benchmark_name)
                #best_found += b
        #best_found /= 10
        #print("\n$$$$$$$$$$$$$$$$$$$$$$$$$best value = ",best_value,"mean best found = ", best_found,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
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