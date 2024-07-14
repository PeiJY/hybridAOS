from __future__ import division
import numpy as np
from numpy.random import rand
import gym
from gym import spaces
from gym.utils import seeding
import math
from scipy.spatial import distance
import time
from scipy.stats import rankdata
from collections import Counter
import os
import random
import math
import json
import copy
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from env.stateless_aos import *
import tensorflow as tf
# ----------- EA components ------------

def O_chg_in_base(route,m,n,co1,co2):
    if co1 < co2:
        seg1 = route[:co1]
        seg2 = route[co1+m:co2+m]
        seg3 = route[co2+m+n:]
        selected_segA = route[co1:co1+m]
        selected_segB = route[co2+m:co2+m+n]
        new_route = seg1 + selected_segB + seg2 + selected_segA + seg3
    else: # . . [n] . . [m] . . 
        seg1 = route[: co2]
        seg2 = route[co2+n: co1+n]
        seg3 = route[co1+n+m:]
        selected_segA = route[co2:co2+n]
        selected_segB = route[co1+n:co1+n+m]
        new_route = seg1 + selected_segB + seg2 + selected_segA + seg3
    return new_route


def O_chg_in_random(individual,m,n):
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>(m+n))[0]
    if len(suitable_route_indexes) == 0: 
        print("o_chg_in: no suitbale route")
        return None
    route_index = suitable_route_indexes[random.randint(0,len(suitable_route_indexes)-1)]
    route = copy.deepcopy(individual[route_index])
    customer_orders = random.sample(range(len(route)-m-n+2),2)
    new_route = O_chg_in_base(route,m,n,customer_orders[0],customer_orders[1])
    individual[route_index] = new_route
    return individual


def O_chg_in_traverse(individual,m,n,fitness,instance,feasi, first_hit = False, time_limit = 100000):
    st = time.time()
    best_individual = copy.deepcopy(individual)
    best_dist = fitness
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>(m+n))[0]
    stop = False
    better_found = False
    if len(suitable_route_indexes) == 0: 
        print("o_chg_in: no suitbale route")
        return None
    for route_index in suitable_route_indexes:
        route = copy.deepcopy(individual[route_index])
        for i in range(len(route)):
            for j in range(len(route)):
                if i == j: continue
                if time.time()-st >=time_limit:
                    return best_individual, best_dist,feasi,better_found
                new_route = O_chg_in_base(route,m,n,i,j)
                t_individual = copy.deepcopy(individual)
                tw_feasible,info,wait_time = instance.route_tw_check(new_route)
                if tw_feasible:
                    demand_feasible = instance.route_demand_check(new_route)
                    if demand_feasible:
                        t_individual[route_index] = new_route
                        dist,feasible,info = instance.evaluate(t_individual)
                        if feasible and dist < best_dist:
                            best_individual = t_individual
                            better_found = True
                            best_dist = dist
                            feasi = True
                            if first_hit:
                               return best_individual, best_dist,feasi,better_found
    return best_individual, best_dist,feasi,better_found

def O_chg_bw_base(route1,route2,m,n,co1,co2):
    seg1 = route1[:co1]
    selected_segA = route1[co1:co1+m]
    seg2 = route1[co1+m:]
    seg3 = route2[:co2]
    selected_segB = route2[co2:co2+n]
    seg4 = route2[co2+n:]
    new_route1 = seg1 + selected_segB + seg2
    new_route2 =seg3 + selected_segA + seg4
    return new_route1,new_route2

def O_chg_bw_random(individual,m,n):
    route_sizes = [len(route) for route in individual]
    suitable_route_indexesA = np.where(np.array(route_sizes)>(m))[0]
    suitable_route_indexesB = np.where(np.array(route_sizes)>(n))[0]
    if len(suitable_route_indexesA) == 0 or len(suitable_route_indexesB) == 0:
        print('o_chg_bw: no suitable route')
        return None
    if len(suitable_route_indexesA) == 1 and len(suitable_route_indexesB) == 1 and suitable_route_indexesA[0] == suitable_route_indexesB[0]:
        print('o_chg_bw: no 2 suitable routes')
        return None
    if len(suitable_route_indexesA) > len(suitable_route_indexesB): # select route from smaller set first
        route_index_n = suitable_route_indexesB[random.randint(0,len(suitable_route_indexesB)-1)]
        route_index_m = suitable_route_indexesA[random.randint(0,len(suitable_route_indexesA)-1)]
        while route_index_n == route_index_m:
            route_index_m = suitable_route_indexesA[random.randint(0,len(suitable_route_indexesA)-1)]
    else:
        route_index_m = suitable_route_indexesA[random.randint(0,len(suitable_route_indexesA)-1)]
        route_index_n = suitable_route_indexesB[random.randint(0,len(suitable_route_indexesB)-1)]
        while route_index_n == route_index_m:
            route_index_n = suitable_route_indexesB[random.randint(0,len(suitable_route_indexesB)-1)]
    selected_route_m = copy.deepcopy(individual[route_index_m])
    customer_order1 = random.randint(0,len(selected_route_m)-m)

    selected_route_n = copy.deepcopy(individual[route_index_n])
    customer_order2 = random.randint(0,len(selected_route_n)-n)

    new_route_m, new_route_n = O_chg_bw_base(selected_route_m,selected_route_n,m,n,customer_order1,customer_order2)

    individual[route_index_m] = new_route_m
    individual[route_index_n] = new_route_n
    return individual

def O_chg_bw_traverse(individual,m,n,fitness,instance,feasi,first_hit = False, time_limit = 100000):
    st = time.time()
    best_individual = copy.deepcopy(individual)
    best_dist = fitness
    stop = False
    better_found = False
    route_sizes = [len(route) for route in individual]
    suitable_route_indexesA = np.where(np.array(route_sizes)>(m))[0]
    suitable_route_indexesB = np.where(np.array(route_sizes)>(n))[0]
    if len(suitable_route_indexesA) == 0 or len(suitable_route_indexesB) == 0:
        print('o_chg_bw: no suitable route')
        return None
    if len(suitable_route_indexesA) == 1 and len(suitable_route_indexesB) == 1 and suitable_route_indexesA[0] == suitable_route_indexesB[0]:
        print('o_chg_bw: no 2 suitable routes')
        return None
    for riA in suitable_route_indexesA:
        for riB in suitable_route_indexesB:
            if riA == riB:continue

            selected_route_m = copy.deepcopy(individual[riA])
            selected_route_n = copy.deepcopy(individual[riB])
            for co1 in range(len(selected_route_m)):
                for co2 in range(len(selected_route_n)):
                    if time.time()-st>=time_limit:
                        return best_individual, best_dist,feasi,better_found
                    new_route_1, new_route_2 = O_chg_bw_base(selected_route_m,selected_route_n,m,n,co1,co2)
                    t_individual = copy.deepcopy(individual)
                    tw_feasible1,info,wait_time = instance.route_tw_check(new_route_1)
                    tw_feasible2,info,wait_time = instance.route_tw_check(new_route_2)
                    if tw_feasible1 and tw_feasible2:
                        if instance.route_demand_check(new_route_1) and instance.route_demand_check(new_route_2):
                            t_individual[riA] = new_route_1
                            t_individual[riB] = new_route_2
                            dist,solution_f,info = instance.evaluate(t_individual)
                            if solution_f and dist < best_dist:
                                best_individual = t_individual
                                better_found =True
                                best_dist = dist
                                feasi = True
                                if first_hit:
                                    return best_individual, best_dist,feasi,better_found
    return best_individual,best_dist,feasi,better_found


def O_ins_in_traverse(individual,m,fitness,instance,feasi,first_hit=False, time_limit = 100000):
    return O_chg_in_traverse(individual,m,0,fitness,instance,feasi,first_hit, time_limit)

def O_ins_in_random(individual,m):
    return O_chg_in_random(individual,m,0)

def O_ins_bw_traverse(individual,m,fitness,instance,feasi,first_hit=False, time_limit = 100000):
    return O_chg_bw_traverse(individual,m,0,fitness,instance,feasi,first_hit,time_limit)

def O_ins_bw_random(individual,m):
    return O_chg_bw_random(individual,m,0)

def O_ruin_recreat_base(indiv, d_rate, instance, route_index, base_customer_index):
    individual = copy.deepcopy(indiv)
    route = copy.deepcopy(individual[route_index])
    removed_customers = []
    temp_route = route
    sum_demand_to_insert = 0
    for customer_index in temp_route:
        if instance.customer_dist_m[customer_index][base_customer_index] < instance.max_customer_dist_l[base_customer_index]*d_rate:
            removed_customers.append(customer_index)
            route.remove(customer_index)
            sum_demand_to_insert += instance.customers[customer_index]['demand']
    individual[route_index] = route
    # try to insert into each other route
    min_wt = float('inf')
    min_wt_route_index = -1
    min_wt_inserted_route = []
    for another_route_index in range(len(individual)):
        if another_route_index == route_index:
            continue
        another_route = copy.deepcopy(individual[another_route_index])
        sum_demand = 0
        for i in another_route:
            sum_demand += instance.customers[i]['demand']
        if sum_demand + sum_demand_to_insert > instance.c:
            continue
        for order in range(len(another_route)):
            temp_route = another_route[:order] + removed_customers + another_route[order:]
            tw_f, info, wt = instance.route_tw_check(temp_route)
            if tw_f and wt < min_wt:
                min_wt = wt
                min_wt_route_index = another_route_index
                min_wt_inserted_route = temp_route
    if min_wt_route_index != -1:
        individual[min_wt_route_index] = min_wt_inserted_route
    else:
        individual.append(removed_customers)
    return individual

def O_ruin_recreat_traverse(individual, d_rate, fitness,instance,feasi,first_hit=False,time_limit=100000):
    st = time.time()
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>=2)[0]
    best_indiv = copy.deepcopy(individual)
    better_found = False
    best_dist = fitness
    if len(suitable_route_indexes) == 0: 
        print("o_chg_in: no suitbale route")
        return None
    for route_index in suitable_route_indexes:
        route = copy.deepcopy(individual[route_index])
        for base_customer_index in route:
            if time.time()-st >= time_limit:
                return best_indiv, best_dist,feasi,better_found
            new_indiv = O_ruin_recreat_base(individual, d_rate, instance, route_index, base_customer_index)
            dist,solution_f,info = instance.evaluate(new_indiv)
            if solution_f and dist<best_dist:
                best_dist = dist
                better_found = True
                best_indiv = new_indiv
                feasi = True
                if first_hit:
                    return best_indiv, best_dist,feasi,better_found
    return best_indiv,best_dist,feasi,better_found

def O_ruin_recreat_random(individual, d_rate,instance):
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>=2)[0]
    if len(suitable_route_indexes) == 0: 
        print("o_chg_in: no suitbale route")
        return None
    route_index = random.sample(suitable_route_indexes,1)
    route = copy.deepcopy(individual[route_index])
    base_customer_index = random.sample(route,1)
    new_indiv = O_ruin_recreat_base(individual, d_rate, instance, route_index, base_customer_index)
    return new_indiv

def O_two_opt_base(route,co1,co2):
    seg1 = route[:co1]
    seg2 = route[co1:co2]
    seg2 = seg2[::-1]
    seg3 = route[co2:]
    new_route = seg1 + seg2 + seg3
    return new_route


def O_two_opt_traverse(individual,fitness,instance,feasi,first_hit=False,time_limit=100000):
    st = time.time()
    best_indiv = copy.deepcopy(individual)
    best_dist = fitness
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>=2)[0]
    better_found = False
    if len(suitable_route_indexes) == 0: 
        print("o_chg_in: no suitbale route")
        return None
    for route_index in suitable_route_indexes:
        route = copy.deepcopy(individual[route_index])
        for co1 in range(len(route)):
            for co2 in range(co1,len(route)):
                if time.time()-st >= time_limit:
                    return best_indiv,best_dist,feasi,better_found
                new_route = O_two_opt_base(route,co1,co2)
                tw_f,info,wt = instance.route_tw_check(new_route)
                if tw_f:
                    if instance.route_demand_check(new_route):
                        new_indiv = copy.deepcopy(individual)
                        new_indiv[route_index] = new_route
                        dist, solution_f,info = instance.evaluate(new_indiv) 
                        if solution_f and dist<best_dist:
                            better_found = True
                            best_dist  = dist
                            best_indiv = new_indiv
                            feasi = True
                            if first_hit:
                                return best_indiv,best_dist,feasi,better_found
    return best_indiv,best_dist,feasi,better_found

def O_two_opt_random(individual):
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>=2)[0]
    if len(suitable_route_indexes) == 0: 
        print("o_chg_in: no suitbale route")
        return None
    route_index = random.sample(suitable_route_indexes)
    route = copy.deepcopy(individual[route_index])
    [co1,co2] = random.sample(route,2)
    new_route = O_two_opt_base(route,co1,co2)
    new_indiv = copy.deepcopy(individual)
    new_indiv[route_index] = new_route
    return new_indiv

def O_two_ope_star_base(route1,route2,co1,co2):
    seg1 = route1[:co1]
    seg2 = route1[co2:]
    seg3 = route2[:co2]
    seg4 = route2[co2:]
    new_route1 = seg1+seg4
    new_route2 = seg3+seg2
    return new_route1,new_route2


def O_two_ope_star_traverse(individual,fitness,instance,feasi,first_hit=False,time_limit=100000):
    st = time.time()
    best_indiv = copy.deepcopy(individual)
    best_dist = fitness
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>2)[0]
    better_found = False
    if len(suitable_route_indexes) <2:
        print('O_two_ope_star: no 2 suitable route')
        return None
    for i in range(len(suitable_route_indexes)):
        for j in range(i,len(suitable_route_indexes)):
            route_index1 = suitable_route_indexes[i]
            route_index2 = suitable_route_indexes[j]
            route1 = copy.deepcopy(individual[route_index1])
            route2 = copy.deepcopy(individual[route_index2])
            for co1 in range(len(route1)):
                for co2 in range(len(route2)):
                    if time.time()-st >= time_limit:
                        return best_indiv,best_dist,feasi, better_found
                    new_route1,new_route2 = O_two_ope_star_base(route1,route2,co1,co2)
                    t_individual = copy.deepcopy(individual)
                    tw_feasible1,info,wait_time = instance.route_tw_check(new_route1)
                    tw_feasible2,info,wait_time = instance.route_tw_check(new_route2)
                    if tw_feasible1 and tw_feasible2:
                        if instance.route_demand_check(new_route1) and instance.route_demand_check(new_route2):
                            t_individual[route_index1] = new_route1
                            t_individual[route_index2] = new_route2
                            dist,solution_f,info = instance.evaluate(t_individual)
                            if solution_f and dist < best_dist:
                                best_indiv = t_individual
                                better_found = True
                                best_dist = dist
                                feasi = True
                                if first_hit:
                                    return best_indiv,best_dist,feasi, better_found
    return best_indiv,best_dist,feasi, better_found

def O_two_ope_star_random(individual):
    route_sizes = [len(route) for route in individual]
    suitable_route_indexes = np.where(np.array(route_sizes)>2)[0]
    if len(suitable_route_indexes) <2:
        print('O_two_ope_star: no 2 suitable route')
        return None
    i = random.sample(suitable_route_indexes,2)
    route_index1 = min(suitable_route_indexes)
    route_index2 = max(suitable_route_indexes)
    route1 = copy.deepcopy(individual[route_index1])
    route2 = copy.deepcopy(individual[route_index2])
    co1 = random.randint(0,len(route1)-1)
    co2 = random.randint(0,len(route2)-1)
    new_route1,new_route2 = O_two_ope_star_base(route1,route2,co1,co2)
    t_individual = copy.deepcopy(individual)
    t_individual[route_index1] = new_route1
    t_individual[route_index2] = new_route2
    return t_individual

# bug free
def hr(nPop,instance):
    pop = []
    fitnesses = []
    feasible = []
    for i in range(nPop*1000):
        visited = [False for _ in range(instance.ncustomer)]
        solution = []
        while False in visited:
            route = []
            capacity_left = instance.c
            current_time = instance.depot['earliest']
            to_be_test_customer_indexes = np.where(np.array(visited)==False)[0]
            # print(visited)
            # print(to_be_test_customer_indexes)
            # print(len(to_be_test_customer_indexes),to_be_test_customer_indexes.shape,to_be_test_customer_indexes.size)
            while len(to_be_test_customer_indexes)>0:
                r = random.randint(0, len(to_be_test_customer_indexes)-1)
                # print(to_be_test_customer_indexes)
                next_customer_index = to_be_test_customer_indexes[r]
                # print(next_customer_index)
                # print(instance.customers)
                next_customer = instance.customers[next_customer_index]
                if next_customer['demand'] <= capacity_left and next_customer['latest'] > current_time:
                    temp_time = max(current_time,next_customer['earliest']) + next_customer['cost']
                    if temp_time < instance.depot['latest']: # able to return depot
                        route.append(next_customer_index)
                        capacity_left -= next_customer['demand']
                        current_time = temp_time
                        visited[next_customer_index] = True
                to_be_test_customer_indexes=np.delete(to_be_test_customer_indexes,r)
            solution.append(route)
        if solution_in_set(solution,pop):
            continue
        dist, feasi, info = instance.evaluate(solution)
        # if not feasible:
        #     raise ValueError("Init solution not feasible, "+info)
        pop.append(solution)
        fitnesses.append(dist)
        feasible.append(feasi)
        if len(pop) >= nPop:
            break
    return pop, fitnesses, feasi

# bug free
def hp(nPop,instance):
    pop = []
    fitnesses = []
    feasible = []
    for i in range(nPop*1000):
        visited = [False for _ in range(instance.ncustomer)]
        solution = []
        while False in visited:
            route = []
            capacity_left = instance.c
            current_time = instance.depot['earliest']
            to_be_test_customer_indexes = np.where(np.array(visited)==False)[0]

            r = random.randint(0, len(to_be_test_customer_indexes)-1)
            first_customer_index = to_be_test_customer_indexes[r]
            route.append(first_customer_index)
            visited[first_customer_index]=True
            first_customer = instance.customers[first_customer_index]
            current_time = max(current_time,first_customer['earliest']) + first_customer['cost']
            capacity_left -= first_customer['demand']
            to_be_test_customer_indexes = np.delete(to_be_test_customer_indexes,r)

            while len(to_be_test_customer_indexes)>0:
                # r = random.randint(0, len(to_be_test_customer_indexes)-1)
                # next_customer_index = to_be_test_customer_indexes[r]

                wait_times = [instance.customers[i]['earliest']-current_time for i in to_be_test_customer_indexes]
                ss = [i for i, x in enumerate(wait_times) if x == min(wait_times)]
                s = ss[random.randint(0,len(ss)-1)]
                # s = wait_times.index(min(wait_times))

                next_customer_index = to_be_test_customer_indexes[s]

                next_customer = instance.customers[next_customer_index]
                if next_customer['demand'] <= capacity_left and next_customer['latest'] > current_time:
                    temp_time = max(current_time,next_customer['earliest']) + next_customer['cost']
                    if temp_time < instance.depot['latest']: # able to return depot
                        route.append(next_customer_index)
                        capacity_left -= next_customer['demand']
                        current_time = temp_time
                        visited[next_customer_index] = True
                to_be_test_customer_indexes=np.delete(to_be_test_customer_indexes,s)
            solution.append(route)
        if solution_in_set(solution,pop):
            continue
        dist, feasi, info = instance.evaluate(solution)
        # if not feasible:
        #     raise ValueError("Init solution not feasible, "+info)
        pop.append(solution)
        fitnesses.append(dist)
        feasible.append(feasi)
        if len(pop) >= nPop:
            break
    return pop, fitnesses,feasible

# bug free
def h1(fitnesses, v, smallest=True):
    temp = [[i,fitnesses[i]] for i in range(len(fitnesses))]
    # print(len(temp),v)
    subset = random.sample(temp,v)
    selected_index = 0
    if smallest:
        smallest_fitness = float('inf')
        for t in subset:
            if t[1] < smallest_fitness:
                smallest_fitness = t[1]
                selected_index = t[0]
    else:
        biggest_fitness = 0
        for t in subset:
            if t[1] > biggest_fitness:
                biggest_fitness = t[1]
                selected_index = t[0]
    return selected_index

    
# bug free
def h1_multi(fitnesses, v, mu, replacement = False):
    indexes = []
    for i in range(mu):
        index = h1(fitnesses, v)
        while (not replacement) and (index in indexes) and (v <= len(fitnesses)):
            index = h1(fitnesses, v)
        indexes.append(index)
    return indexes

def h8(parents, parent_fitnesses, offsprings, offspring_fitnesses, nPop):
    all_fitnesses = parent_fitnesses + offspring_fitnesses
    all_solutions = parents + offsprings
    # best_indexes = np.array(all_fitnesses).argsort()[0:nPop].tolist()
    best_indexes = np.array(all_fitnesses).argsort()[:].tolist()
    selected_solutions = []
    selected_fitnesses = []
    for i in best_indexes:
        # if all_solutions[i] in selected_solutions:
        if solution_in_set(all_solutions[i], selected_solutions):
            continue
        selected_solutions.append(all_solutions[i])
        selected_fitnesses.append(all_fitnesses[i])
        if len(selected_solutions) >= nPop:
            break
    return selected_solutions, selected_fitnesses

def terminate(timestep, max_NoT, start_time, max_runtime):
    return (timestep >= max_NoT or time.time()-start_time >= max_runtime)

def solution_equal(A,B):
    for a in A:
        if a not in B:return False
    for b in B:
        if b not in A: return False
    return True

def solution_in_set(S,SET):
    for i in SET:
        if solution_equal(S,i): 
            # print(S)
            # print(i)
            return True
    return False

def remove_duplicate_solution(pop,fitnesses):
    processed_pop = []
    processed_fitnesses = []
    for i in range(len(pop)):
        if not solution_in_set(pop[i],processed_pop):
            processed_pop.append(pop[i])
            processed_fitnesses.append(fitnesses[i])
    return processed_pop,processed_fitnesses
            

# -------------------------------


# -------- RL components --------

def f1(init_fitnesses,fitnesses):
    # return (sum(init_fitnesses)-sum(fitnesses))/sum(init_fitnesses)
    return sum(fitnesses) / sum(init_fitnesses)

def f2(fitnesses):
    return np.std(fitnesses)

def f3(timestep,NoT):
    return timestep/NoT

def f4(init_fitnesses,fitnesses):
    N = len(fitnesses)
    return N*(max(init_fitnesses)-max(fitnesses))/sum(fitnesses)

def f5(instance):
    return instance.nv

def f6(instance):
    return instance.c

def f7(instance):
    return instance.density_tw

def f8(instance):
    return instance.tightness_tw

# bug free
def reward(init_fitnesses,fitnesses, C):
    f1value = f1(init_fitnesses,fitnesses)
    if f1value<=0:
        # return f1value
        # return -10
        return -1
    if f1value > C:
        return -f1value
    else:
        return -f1value - math.log10(f1value)

def take_action(individual, action,instance, m=1,n=1,d_rate=1):
    if action == 0:
        offspring = O_chg_in_random(individual,m,n)
    elif action == 1:
        offspring = O_chg_bw_random(individual,m,n)
    elif action == 2:
        offspring = O_ins_in_random(individual,m)
    elif action == 3:
        offspring = O_ins_bw_random(individual,m)
    elif action == 4:
        offspring = O_ruin_recreat_random(individual,d_rate,instance)
    elif action == 5:
        offspring = O_two_opt_random(individual)
    elif action == 6:
        offspring = O_two_ope_star_random(individual)
    else:
        raise ValueError("selected evolution operator not exist")
    dist,feasible,info = instance.evaluate(offspring)
    return offspring,dist,feasible

# def take_action_with_check(individual, action,instance, m=1,n=1,d_rate=0.1,repeat = 100):
#     p_dist,p_feasible,p_info = instance.evaluate(individual)
#     for r in range(repeat):
#         offspring,dist,feasible,info = take_action(individual,action,instance,m,n,d_rate)
#         if action == 3 and feasible and dist < p_dist:
#             break
#         if action != 3 and feasible:
#             break
#     return offspring,dist,feasible,info


def take_action_traverse(individual, action,instance, fitness,feasi, m=1,n=1,d_rate=0.1,first_hit = False, time_limit = 100000):
    random.shuffle(individual)
    if action == 0:
        offspring,offspring_dist,offspring_feasible,better_found = O_chg_in_traverse(individual,m,n,fitness,instance,feasi,first_hit,time_limit)
    elif action == 1:
        offspring,offspring_dist,offspring_feasible,better_found = O_chg_bw_traverse(individual,m,n,fitness,instance,feasi,first_hit,time_limit)
    elif action == 2:
        offspring,offspring_dist,offspring_feasible,better_found = O_ins_in_traverse(individual,m,fitness,instance,feasi,first_hit,time_limit)
    elif action == 3:
        offspring,offspring_dist,offspring_feasible,better_found = O_ins_bw_traverse(individual,m,fitness,instance,feasi,first_hit,time_limit)
    elif action == 4:
        offspring,offspring_dist,offspring_feasible,better_found = O_ruin_recreat_traverse(individual,d_rate,fitness,instance,feasi,first_hit,time_limit)
    elif action == 5:
        offspring,offspring_dist,offspring_feasible,better_found = O_two_opt_traverse(individual,fitness,instance,feasi,first_hit,time_limit)
    elif action == 6:
        offspring,offspring_dist,offspring_feasible,better_found = O_two_ope_star_traverse(individual,fitness,instance,feasi,first_hit,time_limit)
    else:
        raise ValueError("selected evolution operator not exist")
    return offspring, offspring_dist,offspring_feasible,better_found, fitness-offspring_dist

# bug free
def obtain_state(init_fitnesses, fitnesses, timestep, NoT, instance):
    return [f1(init_fitnesses,fitnesses),f2(fitnesses),f3(timestep,NoT),f4(init_fitnesses,fitnesses),f5(instance),f6(instance),f7(instance),f8(instance)]
# -------------------------------


# ----------- utils -----------

# bug free
class Instance:
    def __init__(self,file_path):
        self.readInstance(file_path)
        density_tw = 0
        tightness_tw = 0
        for c in self.customers:
            tightness_tw += (c['latest'] - c['earliest'])
            if c['cost'] > 0:
                density_tw += 1
        self.density_tw = density_tw/self.ncustomer
        self.tightness_tw = tightness_tw/self.ncustomer


    def readInstance(self, file_path):
        '''
        json solomon instance file from https://github.com/CervEdin/solomon-vrptw-benchmarks?tab=readme-ov-file
        '''
        with open(file_path, 'r') as f:
            json_dict = json.load(f)
        # print(json_dict)
        self.nv = json_dict['vehicle-nr']
        self.c = json_dict['capacity']
        self.name = json_dict['instance']
        self.depot = json_dict['customers'][0]
        self.customers = json_dict['customers'][1:]
        self.ncustomer = len(self.customers)
        self.customer_dist_m = [[0 for i in range(self.ncustomer)] for j in range(self.ncustomer)]
        self.depot_dist_l = [0 for i in range(self.ncustomer)]
        self.max_customer_dist_l = [0 for i in range(self.ncustomer)]
        x_depot = self.depot['x']
        y_depot = self.depot['y']
        for i in range(self.ncustomer):
            x1 = self.customers[i]['x']
            y1 = self.customers[i]['y']
            self.depot_dist_l[i] = math.sqrt(pow(x1-x_depot,2) + pow(y1-y_depot,2))
            for j in range(i,self.ncustomer):
                x2 = self.customers[j]['x']
                y2 = self.customers[j]['y']
                dist = math.sqrt(pow(x1-x2,2) + pow(y1-y2,2))
                self.customer_dist_m[i][j] = dist
                self.customer_dist_m[j][i] = dist
            self.max_customer_dist_l[i] = max(self.customer_dist_m[i])

    # solution: array of route, route: array of visited customer index (start from 0)

    def is_feasible(self, solution):
        for i in range(len(solution)):
            ri = len(solution)-1-i
            if len(solution[ri])==0:
                solution.pop(ri)

        visited = [False for _ in range(self.ncustomer)]
        if len(solution) > self.nv:
            return False,"to many vehicles"
        for route in solution:
            sum_demand = 0 
            for customer_index in route:
                # once visited check
                if visited[customer_index]:
                    return False,"dulicated visiting"
                visited[customer_index] = True
                customer = self.customers[customer_index]
                sum_demand += customer['demand']  
            # capacity check
            if sum_demand > self.c:
                return False, "capacity overflow"
            tw_check,info,wt = self.route_tw_check(route)
            if not tw_check:
                return tw_check, info
        # all visited check
        if False in visited:
            return False, "not all customer visited"
        return True, ""

    def route_tw_check(self, route):
        current_time = self.depot['earliest']
        wait_time = 0
        for customer_index in route:
            customer = self.customers[customer_index]
            # time window check
            if current_time > customer['latest']:
                return False, "arrive after tw" , -1
            wait_time += max(customer['earliest'], current_time) - current_time
            current_time = max(customer['earliest'], current_time)
            current_time += customer['cost']
        # time window check
        if current_time > self.depot['latest']:
            return False,"return after depot's tw", -1
        return True,"",wait_time

    def route_demand_check(self,route):
        sum_demand = 0
        for customer_index in route:
            sum_demand += self.customers[customer_index]['demand']
            if sum_demand > self.c:
                return False
        return True

    def evaluate(self,solution):
        K = len(solution)
        feasible,info = self.is_feasible(solution)
        total_dist = 0
        for route in solution:
            last_x = self.depot['x']
            last_y = self.depot['y']
            for customer_index in route:
                x = self.customers[customer_index]['x']
                y = self.customers[customer_index]['y']
                total_dist += math.sqrt(pow(last_x-x,2) + pow(last_y-y,2))
                last_x = x
                last_y = y
            total_dist += math.sqrt(pow(last_x-self.depot['x'],2) + pow(last_y-self.depot['y'],2))
        return total_dist+1000*K, feasible, info

    def __str__(self):
        return self.name

Rinstances = []
for i in range(9):
    Rinstances.append('./instances/r10'+str(i+1)+'.json')
for i in range(3):
    Rinstances.append('./instances/r11'+str(i)+'.json')
for i in range(9):
    Rinstances.append('./instances/r20'+str(i+1)+'.json')
for i in range(2):
    Rinstances.append('./instances/r21'+str(i)+'.json')

g_folder = "./temp/"

class GSFCVRPTWenv(gym.Env):
    def __init__(self, instance=None, max_NoT=50, max_runtime=600, nPop=100, epsilon=0.01, write_log = False,shuffle_instance = False,\
                online_update = False, check_overwrite_log = True, init_random_seed=0,eps=0.1,folder=None,\
                wu=0.1,wl=0,Alpha=0.01,Beta=0.01,P_max=0.85,use_random = True,use_joint = True,weight_update_rule = 1,award_global_improvement = True ):
        state_upper_bounds = np.array([1,float('inf'),1,float('inf'),float('inf'),float('inf'),1,float('inf')])
        state_lower_bounds = np.array([0,0,0,0,0,0,0,0])
        self.n_ops = 7
        self.n_obs = 8
        self.nPop = nPop
        self.max_NoT = max_NoT
        self.max_runtime = max_runtime
        self.epsilon = epsilon
        self.action_space = spaces.Discrete(self.n_ops)
        self.observation_space = spaces.Box(state_lower_bounds, state_upper_bounds, shape=(8,), dtype = np.float32)
        self.max_gen = max_NoT

        self.repeat_count = 0
        self.instance = instance
        # self.instance = self.instances[self.instance_index]
        # self.Ac = []
        # self.fitnesses = []
        # self.timestep = 0
        # self.init_fitnesses = []
        self.reset_count = 0
        # not mentioned in origin paper
        self.vforh_1 = 10
        self.mu = 50
        self.lbd = 100
        self.replacement_Hse = False
        self.m = 2
        self.n = 2
        self.d_rate = 0.2
        # self.C = 0.2
        self.C = 0.85
        self.ope_time_limit = 0.5
        self.first_hit = False
        self.shuffle_instance = shuffle_instance

        # init stateless
        self.SLaos = stateless_AOS(ope_num=self.n_ops,CA='FIR', OS='AP',wu=wu,wl=wl,Alpha=Alpha,Beta=Beta,P_max=P_max)
        self.init_random_seed = init_random_seed
        self.use_random = use_random
        self.use_joint = use_joint # if False, use just DE-DDQN      
        self.write_log = write_log 
        self.weight_update_rule = weight_update_rule # 1: w increase if improved | 2: w increase dependent on where the used decision from | 3: AOS to select AOS
        self.award_global_improvement = award_global_improvement


        if folder == None:
            folder = g_folder
        if self.write_log and not os.path.exists(folder):
            os.mkdir(folder)
        folder += str(self.instance)    
        if self.write_log and not os.path.exists(folder):
            os.mkdir(folder)
        log_file = folder+'/log'
        if self.use_joint:
            if self.SLaos.wl==1 and self.SLaos.wu==1:
                if self.use_random:
                    log_file += '_random'
                else:
                    log_file += '_pure_less_'+self.SLaos.CA + '_' + self.SLaos.OS +  '_' + str(self.SLaos.Alpha) +'_' + str(self.SLaos.Beta) +  '_' + str(self.SLaos.P_max)
            else:
                if self.SLaos.wl==self.SLaos.wu and not self.weight_update_rule == 3:
                    if self.use_random:
                        log_file += '_jointrandom_fixweight_'+ str(self.SLaos.wl)
                        self.SLaos.OS = 'random'
                    else:
                        log_file += '_joint_fixweight_'+self.SLaos.CA + '_' + self.SLaos.OS  + '_'+ str(self.SLaos.wl) + '_' + str(self.SLaos.Alpha) +'_' + str(self.SLaos.Beta) +  '_' + str(self.SLaos.P_max) 
                else:
                    if self.use_random:
                        log_file += '_jointrandom_'+ str(self.SLaos.wu) + '_'+ str(self.SLaos.wl)
                        self.SLaos.OS = 'random'
                    else:
                        log_file += '_joint_'+self.SLaos.CA + '_' + self.SLaos.OS + '_' + str(self.SLaos.wu) + '_'+ str(self.SLaos.wl) + '_' + str(self.SLaos.Alpha) +'_' + str(self.SLaos.Beta) +  '_' + str(self.SLaos.P_max) 
            if self.weight_update_rule == 2:
                log_file += '_WU2'
            elif self.weight_update_rule == 3:
                log_file += '_aAOSs'
                self.SLaAOSs = stateless_AOS(ope_num=2, CA='FIR', OS='AP',Alpha=0.01,Beta=0.01,P_max=0.9)         
        if online_update:
            log_file += '_online'
            if not eps == 0.3:
                log_file += '_eps'+str(eps)
        self.log_file = log_file
        if self.write_log:
            if check_overwrite_log:
                if os.path.exists(log_file):
                    raise Exception(log_file, 'log file existing')
                f = open(log_file,'w')
                info = self.SLaos.CA + ' ' + self.SLaos.OS + ' ' + str(self.SLaos.wu) + ' '+ str(self.SLaos.wl) + ' ' + str(self.SLaos.Alpha) +' ' + str(self.SLaos.Beta) +  ' ' + str(self.SLaos.P_max)
                f.write( info+ '\n')
                f.close()
            else:
                if not os.path.exists(log_file):
                    f = open(log_file,'a')
                    info = self.SLaos.CA + ' ' + self.SLaos.OS + ' ' + str(self.SLaos.wu) + ' '+ str(self.SLaos.wl) + ' ' + str(self.SLaos.Alpha) +' ' + str(self.SLaos.Beta) +  ' ' + str(self.SLaos.P_max)
                    f.write( info+ '\n')
        print(self.log_file)

    def step(self, action):

        # -------- joint stateless and state-based --------
        if self.use_joint:
            if self.use_random and self.SLaos.wl==1 and self.SLaos.wu==1:
                prob = [1/self.n_ops for _ in range(self.n_ops)]
                action = random_select(prob)
            else:
                if self.weight_update_rule == 1 or self.weight_update_rule == 2:
                    chosen_AOS, action = self.SLaos.joint_select(Sb_action=action)
                elif self.weight_update_rule == 3: # select AOS method with another SLaos
                    chosen_AOS = self.SLaAOSs.select_ope()
                    if chosen_AOS == 0:
                        action = self.SLaos.select_ope()
                else:
                    raise Exception(self.weight_update_rule, 'weight_update_rule not exist')
        # --------------------- end -----------------------   

        # evolution seleciton: select parent
        parent_indexes = h1_multi(self.fitnesses,self.vforh_1, self.mu, self.replacement_Hse)
        parent_fitnesses = []
        parent_feasible = []
        Ap = []
        for i in parent_indexes:
            parent_fitnesses.append(self.fitnesses[i])
            parent_feasible.append(self.feasible[i])
            Ap.append(self.Ac[i])

        # generate offspring 
        Ao = []
        offspring_fitnesses = []
        offspring_feasible = []

        # parallelised operator using and new solution evaluation
        pool_size = len(Ap)
        inputs = []
        stateless_r = 0.
        for _ in range(1):
            if len(Ao) >= self.lbd:
                break
            for x in range(len(Ap)):
                single_input = (Ap[x],action,self.instance,parent_fitnesses[x],parent_feasible[x],self.m,self.n,self.d_rate,self.first_hit,self.ope_time_limit)
                inputs.append(single_input)
            action_pool = Pool(pool_size)
            outputs = action_pool.starmap(take_action_traverse,inputs)
            action_pool.close()
            action_pool.join()
            for single_output in outputs:
                (offspring, offspring_dist,offspring_feasi,better_found,delta_dist) = single_output
                if offspring_feasi and better_found:
                    Ao.append(offspring)
                    offspring_fitnesses.append(offspring_dist)
                    offspring_feasible.append(offspring_feasi) # should be all feasible
                    stateless_r += delta_dist
                if len(Ao) >= self.lbd:
                    break
        if len(Ao) > 0:
            stateless_r /= len(Ao)

        # operator using and new solution evaluation
        # for index in range(len(Ap)*10):
        #     individual = Ap[index%len(Ap)]
        #     fitness = parent_fitnesses[index%len(Ap)]
            
        #     feasi = parent_feasible[index%len(Ap)]
        #     offspring, offspring_dist,offspring_feasi,better_found = take_action_traverse(individual, action,self.instance, fitness,feasi,self.m, self.n, self.d_rate,first_hit=self.first_hit, time_limit=self.ope_time_limit)
        #     # offspring, offspring_dist,offspring_feasi = take_action(individual, action,self.instance,self.m, self.n, self.d_rate)
        #     if offspring_feasi and better_found:
        #         Ao.append(offspring)
        #         offspring_fitnesses.append(offspring_dist)
        #         offspring_feasible.append(offspring_feasi) # should be all feasible
        #     if len(Ao) >= self.lbd:
        #         break
        #     if terminate(self.timestep, self.max_NoT, self.start_time, self.max_runtime):
        #         break

        # replacement selection: select next population 
        # new_Ac, new_fitnesses = h8(Ap, parent_fitnesses, Ao, offspring_fitnesses, self.nPop) # select from selected parent set and offspring set 
        new_Ac, new_fitnesses = h8(self.Ac, self.fitnesses, Ao, offspring_fitnesses, self.nPop)   # select from all curretn solution and offspring set

        # reward calculation for RL
        r = reward( self.init_fitnesses,new_fitnesses, self.C)

        # update stateless AOS
        if self.award_global_improvement:
            if min(new_fitnesses) < min(self.fitnesses):
                    stateless_r *= 10
        if self.use_joint and not self.use_random:
            self.SLaos.update_select_probs(r,action)
            improved = (r>0)
            if self.weight_update_rule == 2:
                self.SLaos.weight_update2(chosen_AOS,improved)
            elif self.weight_update_rule == 3:
                self.SLaAOSs.update_select_probs(r,chosen_AOS)[0]
            else:
                self.SLaos.weight_update(improved)


        # update population
        self.Ac = new_Ac
        self.fitnesses = new_fitnesses

        # obtain next state observation
        next_state_obs = obtain_state(self.init_fitnesses, self.fitnesses, self.timestep,self.max_NoT,self.instance)
        self.timestep += 1
        stop = terminate(self.timestep, self.max_NoT, self.start_time, self.max_runtime)
        print(self.timestep, time.time()-self.start_time, min(self.fitnesses),next_state_obs,action,r)
        if stop:
            if self.write_log:
                logf = open(self.log_file,'a')
                logf.write( str(min(self.fitnesses)) + '\n')
                logf.close()
        if self.use_joint:
            return next_state_obs, r, stop, {}, stateless_r
        return next_state_obs, r, stop, {}



    def reset(self):
        # if self.instance_index == 0 and self.shuffle_instance:
        #     random.shuffle(self.instances)
        # self.instance = self.instances[self.instance_index]
        random.seed(self.init_random_seed+self.reset_count)
        np.random.seed(self.init_random_seed+self.reset_count)
        tf.set_random_seed(self.init_random_seed+self.reset_count)
        self.runtime = 0
        self.timestep = 0
        self.start_time = time.time()
        print('--- reset ', self.reset_count,', seed: ',self.init_random_seed+self.reset_count,'|',self.log_file,' ---')
        
    
        self.reset_count += 1
        
        # reset population 
        self.Ac,self.fitnesses,self.feasible = hp(self.nPop,self.instance)
        self.init_fitnesses = copy.deepcopy(self.fitnesses)
        print(min(self.fitnesses))
        obs = obtain_state(self.init_fitnesses, self.fitnesses, self.timestep,self.max_NoT,self.instance)

        # reset stateless AOS
        self.SLaos.restart_records()
        
        return obs



