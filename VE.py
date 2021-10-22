# -*- coding: utf-8 -*-
"""
@author: ydb
"""

class Factor:
    def __init__(self, s, value_dict):
        self.scope = s
        self.value = value_dict

# The format of the evidence input should be a dictionary with 
# {1:1,2:0}, which means the evidence is that x1 = 1 and x2 = 0.
def reduce_factor(factor, evidence):
    scope = []
    for node in factor.scope:
        if node not in evidence:
            scope.append(node)
    if len(scope) == len(factor.scope):
        return factor
    else:
        value_dict = {}
        for config in factor.value:
            temp_config = []
            for i in range(len(config)):
                if factor.scope[i] in evidence:
                    if config[i] != evidence[factor.scope[i]]:
                        temp_config = []
                        break
                else:
                    temp_config.append(config[i])
            if len(temp_config) != 0:
                value_dict[tuple(temp_config)] = factor.value[config]
    return Factor(scope, value_dict)

# List all possible configurations 
def get_all_configuration(scope):
    n = len(scope)
    queue = [[0],[1]]
    while queue:
        if len(queue[0]) == n:
            break
        else:
            temp_config = queue.pop(0)
            for p in [0,1]:
                queue.append(temp_config + [p])
    res = []
    for config in queue:
        temp_dict = {}
        for i in range(n):
            temp_dict[scope[i]] = config[i]
        res.append(temp_dict)
    return res
# get_all_configuration([2,5,6])

def sum_product_eliminate_var(factors, eliminate_ordering):
    for node in eliminate_ordering:
        factors = VE(factors,node)
    return factors

def VE(factors,eliminate_node):
    res_factors = []
    involved_factor = []
    for factor in factors:
        if eliminate_node in factor.scope:
            involved_factor.append(factor)
        else:
            res_factors.append(factor)
    new_scope = set()
    for factor in involved_factor:
        for node in factor.scope:
            if node != eliminate_node:
                new_scope.add(node)
    new_scope = list(new_scope)
    value_dict = {}
    config_list = get_all_configuration(new_scope)
    for config in config_list:
        temp_sum = 0
        for eliminate_config in [0,1]:
            temp_product = 1
            for factor in involved_factor:
                temp_config = []
                for node in factor.scope:
                    if node == eliminate_node:
                        temp_config.append(eliminate_config)
                    else:
                        temp_config.append(config[node])
                if tuple(temp_config) in factor.value:
                    temp_product = temp_product * factor.value[tuple(temp_config)]
                else:
                    temp_product = 0 
            temp_sum = temp_sum + temp_product
        value_dict[tuple(config.values())] = temp_sum
    new_factor = Factor(new_scope, value_dict = value_dict)
    res_factors.append(new_factor)
    return res_factors

# Input the known factors fron question 1.
phi1 = Factor([1,2],{(0,0):13, (0,1):14, (1,0):11, (1,1):1})
phi2 = Factor([1,3],{(0,0):15, (0,1):2,  (1,0):16, (1,1):6})
phi3 = Factor([2,4],{(0,0):4,  (0,1):8,  (1,0):2,  (1,1):8})
phi4 = Factor([3,5],{(0,0):1,  (0,1):5,  (1,0):15, (1,1):15})
phi5 = Factor([5,6],{(0,0):7,  (0,1):19, (1,0):7,  (1,1):2})
phi6 = Factor([2,6],{(0,0):9,  (0,1):10, (1,0):16, (1,1):1})
phi = []
phi.append(phi1)
phi.append(phi2)
phi.append(phi3)
phi.append(phi4)
phi.append(phi5)
phi.append(phi6)

# Question c and d.
eliminate_ordering = [6,5,4,3,2]
final_result = sum_product_eliminate_var(phi, eliminate_ordering)
print(final_result[0].scope)
print(final_result[0].value)

# Another way of calculating normalizing constant.
all_configs = get_all_configuration([1,2,3,4,5,6])
res = 0 
for config in all_configs:
    temp = 1
    for factor in phi:
        temp_config = []
        for node in factor.scope:
            temp_config.append(config[node])
        temp = temp * factor.value[tuple(temp_config)]
    res = res + temp
print(res)
print(final_result[0].value[(1,)]/res)

# Question e.
evidence = {5:1}
reduced_factors = []
for factor in phi:
    reduced_factors.append(reduce_factor(factor, evidence))
# Find the normalizing constant.
all_configs2 = get_all_configuration([1,2,3,4,6])
res = 0 
for config in all_configs2:
    temp = 1
    for factor in reduced_factors:
        temp_config = []
        for node in factor.scope:
            temp_config.append(config[node])
        temp = temp * factor.value[tuple(temp_config)]
    res = res + temp
print(res)

# Get the conditional probability
eliminate_ordering2 = [6,4,3,2]
final_result2 = sum_product_eliminate_var(reduced_factors, eliminate_ordering2)
print(len(final_result2))
print(final_result2[0].scope)
print(final_result2[0].value)
print(final_result2[1].scope)
print(final_result2[1].value)

a = final_result2[0].value[(0,)] * final_result2[1].value[(0,)] + final_result2[0].value[(1,)] * final_result2[1].value[(1,)]
print(a)
print((final_result2[0].value[(1,)] * final_result2[1].value[(1,)])/a)


