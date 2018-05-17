# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
# from l1 import l1
from cvxopt import matrix, solvers
import numpy as np
"""
0: up
1: down
2: left
3: right
"""

from cal_plot import plot_map, cal_state_val, cal_optimal_policy, ini_P_matrix, ini_R_matrix
Actions = {0,1,2,3}

Df = 0.8
eps = 0.01
w = 0.1

P=ini_P_matrix()
R1, R2 = ini_R_matrix()

# plot_map(R1,0)
# plot_map(R2,0)

V1=cal_state_val(R1,P,Df,eps)
V2=cal_state_val(R2,P,Df,eps)  

# plot_map(V1,0)
# plot_map(V2,0)

pi1=cal_optimal_policy(V1,R1,P,Df)
pi2=cal_optimal_policy(V2,R2,P,Df)


# plot_map(pi1,1)
# plot_map(pi2,1)

# R = np.zeros((10,10))

def Reward(lamda, Rmax):
    # c = matrix(np.concatenate((np.zeros(100,), np.ones(100,), np.full((100, ), -lamda))).astype(np.double))
    c = matrix(-np.hstack([np.zeros((100,)), np.ones((100,)), -lamda*np.ones((100,))]))
    Pa1 = np.zeros((100, 100))
    Pa = np.zeros((100, 100))
    P_long = []
    I = np.eye(100)
    Z = np.zeros((100,100))

    # for s in range(100):
    #     # r = s%10 
    #     # c = s//10
    #     opti_policy = int(pi1[s%10, s//10])
    #     Pa1[s, :] = P[opti_policy, s, :]
    #     for a in A - {opti_policy}:
    #         P_long.append(np.dot(P[opti_policy, s, :] - P[a, s, :], np.linalg.inv(I-0.8*P[opti_policy, s, :])))


    # P_long = -np.dot((Pa1-Pa), np.linalg.inv(I-0.8*Pa1))

    # P_long = np.vstack(P_long)
    # print len(P_long)

    #This part is from https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/linear_irl.py
    ##not sure how to write 
    def T(a, s):
            opti_policy = int(pi1[s%10, s//10])
            return np.dot(P[opti_policy, s, :] - P[a, s, :], np.linalg.inv(I-0.8*P[opti_policy, s, :]))

    Z2 = np.zeros((100*(4-1), 100)) #300*100
    P_long = np.vstack([
            -T(a, s)
            for s in range(100)
            for a in Actions - {int(pi1[s%10, s//10])}
        ])
    I2 = np.vstack([
            np.eye(1, 100, s)
            for s in range(100)
            for a in Actions - {int(pi1[s%10, s//10])}
    ])

    ####

    # print I2.size

    R_para = np.vstack([P_long, P_long, -I, I, I, -I])
    ti_para = np.vstack([I2, Z2, Z, Z, Z, Z])
    u_para = np.vstack([Z2, Z2, -I, -I, Z, Z])

    # print R_para.size
    # print ti_para.size
    # print u_para.size

    A = matrix(np.hstack([R_para, ti_para, u_para]))
    b = matrix(np.concatenate((np.zeros(8*100,), np.full((2*100, ), Rmax))).astype(np.double))

    # b = matrix(np.concatenate(np.zeros((600, 1)), np.full((2*100, ), 1))).astype(np.double)))
    # b = matrix(np.concatenate((np.zeros(*100,), np.full((2*100, ), 1))).astype(np.double))
    # print A.size
    # print b.size
    # print c.size


    sol = solvers.lp(c, A, b)

    #R_new = np.reshape(sol['x'][:100], (10,10))
    R_new = sol['x'][:100]
    R_new_10x10 = np.zeros((10,10))
    count = 0
    for x in range(10):
        for y in range(10):
            R_new_10x10[y][x] = R_new[count]
            count+=1

    return R_new_10x10

def Acc(R):
    V_inv = cal_state_val(R,P,Df,eps)
    poli_inv = cal_optimal_policy(V_inv, R, P, Df)
    # plot_map(poli_inv, 1)
    return np.sum(poli_inv == pi1)/100.0

"""
Q11
"""
lamdas = np.arange(0.0, 5.01, 1) ## 1 should be changed to 0.01
accs = []


for lamda in lamdas:
    accs.append(Acc(Reward(lamda, 1)))

# print accs
plt.plot(lamdas, accs)
plt.show()

"""
Q12
"""
max_lamda = lamdas[accs.index(max(accs))]
print max_lamda

"""
Q13
"""
#ground truth
plot_map(R1,0)
#extracted reward
plot_map(Reward(max_lamda, 1),0)

"""
Q14
"""
value=cal_state_val(Reward(max_lamda, 1),P,Df,eps)
plot_map(value,0)

"""
Q15
"""
#Compare the heat maps of Question 3 and Question 14

"""
Q16
"""
po=cal_optimal_policy(value, Reward(max_lamda, 1), P, Df)
plot_map(po,1)

"""
Q17
"""
#Compare the figures of Question 5 and Question 16
