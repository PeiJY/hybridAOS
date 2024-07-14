import numpy as np
import math
# import validate_dqn_w_log as validate_dqn
import validate_dqn
import random
import time
from env.GSFCVRPTW_test import *

Rinstances = []
for i in range(9):
    Rinstances.append('./env/instances/r10'+str(i+1)+'.json')
for i in range(3):
    Rinstances.append('./env/instances/r11'+str(i)+'.json')
for i in range(9):
    Rinstances.append('./env/instances/r20'+str(i+1)+'.json')
for i in range(2):
    Rinstances.append('./env/instances/r21'+str(i)+'.json')

Cinstances1 = []
Cinstances2 = []
for i in range(9):
    Cinstances1.append('./env/instances/c10'+str(i+1)+'.json')
for i in range(8):
    Cinstances2.append('./env/instances/c20'+str(i+1)+'.json')

RCinstances1 = []
RCinstances2 = []
for i in range(8):
    RCinstances1.append('./env/instances/rc10'+str(i+1)+'.json')
for i in range(8):
    RCinstances2.append('./env/instances/rc20'+str(i+1)+'.json')

# ALLinstances = Rinstances + Cinstances + RCinstances
# ALLinstances = RCinstances2
# ALLinstances = [Rinstances[0]]
# ALLinstances = [Cinstances1[0]]
# ALLinstances = ['./env/instances/c203.json']
ALLinstances = Rinstances[1:] 
print(ALLinstances)
wu=0.1
wl=0.5
Alpha=0.01
Beta=0.01
P_max=0.85
# Alpha=0.1
# Beta=0.1
# P_max=0.5
use_joint = True # apply stateless on state-based or not
use_random = False # is stateless random or not
write_log = True
award_global_improvement = True
weight_update_rule = 1
folder  = './test_log/'


st = time.time()
seeds = random.sample(range(0, 10000), 30)
random.seed(seeds[0])
np.random.seed(seeds[0])


for j in range(len(ALLinstances)):
    ins = Instance(ALLinstances[j])
    # validate_dqn.GSF_online_updated(ins, lbounds, ubounds, best_value,benchmark_name,random_seeds=seeds,folder='./temp2/',eps=0.3,\
    #                                 wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
    #                                 award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)

    # validate_dqn.GSF(ins,wu=wu,wl=wl,folder=folder,\
    #                  Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
    #                                 award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)
    

    validate_dqn.GSF_online_updated(ins,wu=wu,wl=wl,folder=folder,random_seeds=seeds,eps=0.3,\
                     Alpha=Alpha,Beta=Beta,P_max=P_max,use_joint=use_joint,use_random=use_random,\
                                    award_global_improvement=award_global_improvement,weight_update_rule=weight_update_rule)

print("run time (hour): ", (time.time()-st)/3600)
