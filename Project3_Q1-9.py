# -*- coding: utf-8 -*-
"""
Created on Fri May 11 20:40:20 2018

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np
"""
0: up
1: down
2: left
3: right
"""
def plot_map(V,policy):
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.xticks(np.arange(0,11,1))
    plt.yticks(np.arange(0,11,1))
    if policy == 0:  
        plt.pcolor(V)
        plt.colorbar()  
        for x in np.arange(0,V.shape[0],1):
            for y in np.arange(0,V.shape[1],1):
                plt.gca().text(y+0.5,x+0.5,format(V[x, y], '.1f'),horizontalalignment="center")
    else:
        arrow = [u'\u2191', u'\u2193', u'\u2190', u'\u2192']
        for x in np.arange(0,V.shape[0],1):
            for y in np.arange(0,V.shape[1],1):
                plt.gca().text(y+0.5,x+0.5,arrow[int(V[x,y])])
    plt.show()

def cal_state_val(R,P,Df,eps):
    V=np.zeros((10,10))
    delta = float('inf')
    while delta > eps:
        delta = 0
        for i in range(0,10):
            for j in range(0,10):
                v = V[i,j]
                candidate = np.zeros(4)
                for k in range(4):
                    if (i,j) not in ((0,0),(0,9),(9,0),(9,9)):
                        candidate[k] += P[k,j*10+i,j*10+max((i-1),0)]*(R[max((i-1),0),j]+Df*V[max((i-1),0),j]) +\
                                        P[k,j*10+i,j*10+min((i+1),9)]*(R[min((i+1),9),j]+Df*V[min((i+1),9),j]) +\
                                        P[k,j*10+i,max((j-1),0)*10+i]*(R[i,max((j-1),0)]+Df*V[i,max((j-1),0)]) +\
                                        P[k,j*10+i,min((j+1),9)*10+i]*(R[i,min((j+1),9)]+Df*V[i,min((j+1),9)])
                    else:
                        candidate[k] += P[k,j*10+i,j*10+i]*(R[i,j]+Df*V[i,j])
                        if i==0 and j==0: candidate[k] += P[k,j*10+i,j*10+(i+1)]*(R[i+1,j]+Df*V[i+1,j]) + P[k,j*10+i,(j+1)*10+i]*(R[i,j+1]+Df*V[i,j+1])
                        elif i==0 and j==9: candidate[k] += P[k,j*10+i,(j-1)*10+i]*(R[i,j-1]+Df*V[i,j-1]) + P[k,j*10+i,j*10+(i+1)]*(R[i+1,j]+Df*V[i+1,j])
                        elif i==9 and j==0: candidate[k] += P[k,j*10+i,j*10+(i-1)]*(R[i-1,j]+Df*V[i-1,j]) + P[k,j*10+i,(j+1)*10+i]*(R[i,j+1]+Df*V[i,j+1])
                        elif i==9 and j==9: candidate[k] += P[k,j*10+i,(j-1)*10+i]*(R[i,j-1]+Df*V[i,j-1]) + P[k,j*10+i,j*10+(i-1)]*(R[i-1,j]+Df*V[i-1,j])      
                V[i,j] = np.amax(candidate)
                delta = max(delta, abs(v-V[i,j]))
    return V

def cal_optimal_policy(V):
    pi=np.zeros((10,10))
    for i in range(0,10):
        for j in range(0,10):
            temp = V[i,j]
            if temp <= V[max((i-1),0),j]:
                temp =  V[max((i-1),0),j]
                pi[i,j] = 0
            if temp <= V[min((i+1),9),j]:
                temp =  V[min((i+1),9),j]
                pi[i,j] = 1
            if temp <= V[i,max((j-1),0)]:
                temp =  V[i,max((j-1),0)]
                pi[i,j] = 2
            if temp <= V[i,min((j+1),9)]:
                temp =  V[i,min((j+1),9)]
                pi[i,j] = 3
    return pi

def ini_P_matrix():
    P=np.zeros((4,100,100))
    ################
    P[0,9,8] = 1 - w + w/4
    P[0,9,9] = w/4 + w/4
    P[0,9,19] = w/4
    
    P[1,9,8] = w/4
    P[1,9,9] = 1 - w + w/4 + w/4
    P[1,9,19] = w/4
    
    P[2,9,8] = w/4
    P[2,9,9] = 1 - w + w/4 + w/4
    P[2,9,19] = w/4
    
    P[3,9,8] = w/4
    P[3,9,9] = w/4 + w/4
    P[3,9,19] = 1 - w + w/4
    ################
    ################
    P[0,0,0] = 1 - w + w/4 + w/4
    P[0,0,1] = w/4
    P[0,0,10] = w/4
    
    P[1,0,0] = w/4 + w/4
    P[1,0,1] = 1 - w + w/4
    P[1,0,10] = w/4
    
    P[2,0,0] = 1 - w + w/4 + w/4
    P[2,0,1] = w/4
    P[2,0,10] = w/4
    
    P[3,0,0] = w/4 + w/4
    P[3,0,1] = w/4
    P[3,0,10] = 1 - w + w/4
    ################
    ################
    P[0,90,90] = 1 - w + w/4 + w/4
    P[0,90,91] = w/4
    P[0,90,80] = w/4
    
    P[1,90,90] = w/4 + w/4
    P[1,90,91] = 1 - w + w/4
    P[1,90,80] = w/4
    
    P[2,90,90] = w/4 + w/4
    P[2,90,91] = w/4
    P[2,90,80] = 1 - w + w/4
    
    P[3,90,90] = 1 - w + w/4 + w/4
    P[3,90,91] = w/4
    P[3,90,80] = w/4
    ################
    ################
    P[0,99,98] = 1 - w + w/4 
    P[0,99,99] = w/4 + w/4
    P[0,99,89] = w/4
    
    P[1,99,98] = w/4 
    P[1,99,99] = 1 - w + w/4 + w/4
    P[1,99,89] = w/4
    
    P[2,99,98] = w/4 
    P[2,99,99] = w/4 + w/4
    P[2,99,89] = 1 - w + w/4
    
    P[3,99,98] = w/4 
    P[3,99,99] = 1 - w + w/4 + w/4
    P[3,99,89] = w/4
    ################
    
    for i in range(0,10):
        for j in range(0,10):
            if (i,j) not in ((0,0),(0,9),(9,0),(9,9)):
                P[0,i*10+j,i*10+max((j-1),0)] = 1 - w + w/4
                P[0,i*10+j,i*10+min((j+1),9)] = w/4
                P[0,i*10+j,max((i-1),0)*10+j] = w/4
                P[0,i*10+j,min((i+1),9)*10+j] = w/4
            
                P[1,i*10+j,i*10+max((j-1),0)] = w/4
                P[1,i*10+j,i*10+min((j+1),9)] = 1 - w + w/4
                P[1,i*10+j,max((i-1),0)*10+j] = w/4
                P[1,i*10+j,min((i+1),9)*10+j] = w/4
            
                P[2,i*10+j,i*10+max((j-1),0)] = w/4
                P[2,i*10+j,i*10+min((j+1),9)] = w/4
                P[2,i*10+j,max((i-1),0)*10+j] = 1 - w + w/4
                P[2,i*10+j,min((i+1),9)*10+j] = w/4
            
                P[3,i*10+j,i*10+max((j-1),0)] = w/4
                P[3,i*10+j,i*10+min((j+1),9)] = w/4
                P[3,i*10+j,max((i-1),0)*10+j] = w/4
                P[3,i*10+j,min((i+1),9)*10+j] = 1 - w + w/4
    return P

def ini_R_matrix():
    R1=np.zeros((10,10))
    R2=np.zeros((10,10))
    ################
    R1[9,9]=1
    R2[1:7,4]=-100
    R2[1,5]=-100
    R2[1:4,6]=-100
    R2[7:9,6]=-100
    R2[3,7]=-100
    R2[7,7]=-100
    R2[3:8,8]=-100
    R2[9,9]=10
    ################
    return R1, R2
Df = 0.8
eps = 0.01
w = 0.1

P=ini_P_matrix()
R1, R2 = ini_R_matrix()

plot_map(R1,0)
plot_map(R2,0)

V1=cal_state_val(R1,P,Df,eps)
V2=cal_state_val(R2,P,Df,eps)  

plot_map(V1,0)
plot_map(V2,0)

pi1=cal_optimal_policy(V1)
pi2=cal_optimal_policy(V2)

plot_map(pi1,1)
plot_map(pi2,1)


