######################
# Author: Te-Yuan Liu
######################

######################
# Import library
######################

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
######################
# Define function
######################
def init_P(_w):
    # 0:up, 1:down, 2:left, 3:right
    w = _w
    P = np.zeros((4,100,100))

    P[0,0,0] = 1 - w + w/4 + w/4
    P[0,0,1] = w/4
    P[0,0,10] = w/4

    P[0,9,8] = 1 - w + w/4
    P[0,9,9] = w/4 + w/4
    P[0,9,19] = w/4

    P[0,90,90] = 1 - w + w/4 + w/4
    P[0,90,91] = w/4
    P[0,90,80] = w/4

    P[0,99,98] = 1 - w + w/4
    P[0,99,99] = w/4 + w/4
    P[0,99,89] = w/4

    P[1,0,0] = w/4 + w/4
    P[1,0,1] = 1 - w + w/4
    P[1,0,10] = w/4

    P[1,9,8] = w/4
    P[1,9,9] = 1 - w + w/4 + w/4
    P[1,9,19] = w/4

    P[1,90,90] = w/4 + w/4
    P[1,90,91] = 1 - w + w/4
    P[1,90,80] = w/4

    P[1,99,98] = w/4
    P[1,99,99] = 1 - w + w/4 + w/4
    P[1,99,89] = w/4

    P[2,0,0] = 1 - w + w/4 + w/4
    P[2,0,1] = w/4
    P[2,0,10] = w/4

    P[2,9,8] = w/4
    P[2,9,9] = 1 - w + w/4 + w/4
    P[2,9,19] = w/4

    P[2,90,90] = w/4 + w/4
    P[2,90,91] = w/4
    P[2,90,80] = 1 - w + w/4

    P[2,99,98] = w/4
    P[2,99,99] = w/4 + w/4
    P[2,99,89] = 1 - w + w/4

    P[3,0,0] = w/4 + w/4
    P[3,0,1] = w/4
    P[3,0,10] = 1 - w + w/4

    P[3,9,8] = w/4
    P[3,9,9] = w/4 + w/4
    P[3,9,19] = 1 - w + w/4

    P[3,90,90] = 1 - w + w/4 + w/4
    P[3,90,91] = w/4
    P[3,90,80] = w/4

    P[3,99,98] = w/4
    P[3,99,99] = 1 - w + w/4 + w/4
    P[3,99,89] = w/4
    
    for i in range(10):
        for j in range(10):
            if (i,j) not in ((0,0),(0,9),(9,0),(9,9)):
                P[0,i*10+j,i*10+max(j-1,0)] = 1 - w + w/4
                P[0,i*10+j,i*10+min(j+1,9)] = w/4
                P[0,i*10+j,max(i-1,0)*10+j] = w/4
                P[0,i*10+j,min(i+1,9)*10+j] = w/4

                P[1,i*10+j,i*10+max(j-1,0)] = w/4
                P[1,i*10+j,i*10+min(j+1,9)] = 1 - w + w/4
                P[1,i*10+j,max(i-1,0)*10+j] = w/4
                P[1,i*10+j,min(i+1,9)*10+j] = w/4

                P[2,i*10+j,i*10+max(j-1,0)] = w/4
                P[2,i*10+j,i*10+min(j+1,9)] = w/4
                P[2,i*10+j,max(i-1,0)*10+j] = 1 - w + w/4
                P[2,i*10+j,min(i+1,9)*10+j] = w/4

                P[3,i*10+j,i*10+max(j-1,0)] = w/4
                P[3,i*10+j,i*10+min(j+1,9)] = w/4
                P[3,i*10+j,max(i-1,0)*10+j] = w/4
                P[3,i*10+j,min(i+1,9)*10+j] = 1 - w + w/4
    return P

def init_R():
    R1 = np.zeros((10,10))
    R1[9,9] = 1

    R2 = np.zeros((10,10))
    R2[9,9] = 10
    R2[1:7,4] = -100
    R2[1,5:7] = -100
    R2[2:4,6] = -100
    R2[3,7:9] = -100
    R2[4:8,8] = -100
    R2[7,6:8] = -100
    R2[8,6] = -100

    return R1,R2

def compute_pi(P, R, r, e):
    V = np.zeros((10,10))
    Pi = np.zeros((10,10))
    delta = 2*e
    while delta > e:
        delta = 0
        for i in range(10):
            for j in range(10):
                v = V[i,j]
                A = np.zeros(4)
                for k in range(4):
                    if (i,j) not in ((0,0),(0,9),(9,0),(9,9)):
                        A[k] = P[k,j*10+i,j*10+max(i-1,0)]*(R[max(i-1,0),j]+r*V[max(i-1,0),j])+\
                            P[k,j*10+i,j*10+min(i+1,9)]*(R[min(i+1,9),j]+r*V[min(i+1,9),j])+\
                            P[k,j*10+i,max(j-1,0)*10+i]*(R[i,max(j-1,0)]+r*V[i,max(j-1,0)])+\
                            P[k,j*10+i,min(j+1,9)*10+i]*(R[i,min(j+1,9)]+r*V[i,min(j+1,9)])
                    else:
                        A[k] = P[k,j*10+i,j*10+i]*(R[i,j]+r*V[i,j])
                        if i==0 and j==0: A[k] += P[k,j*10+i,j*10+i+1]*(R[i+1,j]+r*V[i+1,j]) + P[k,j*10+i,(j+1)*10+i]*(R[i,j+1]+r*V[i,j+1])
                        elif i==9 and j==0: A[k] += P[k,j*10+i,j*10+i-1]*(R[i-1,j]+r*V[i-1,j]) + P[k,j*10+i,(j+1)*10+i]*(R[i,j+1]+r*V[i,j+1])
                        elif i==0 and j==9: A[k] += P[k,j*10+i,j*10+i+1]*(R[i+1,j]+r*V[i+1,j]) + P[k,j*10+i,(j-1)*10+i]*(R[i,j-1]+r*V[i,j-1])
                        elif i==9 and j==9: A[k] += P[k,j*10+i,j*10+i-1]*(R[i-1,j]+r*V[i-1,j]) + P[k,j*10+i,(j-1)*10+i]*(R[i,j-1]+r*V[i,j-1])
                V[i,j] = np.amax(A)
                delta = max(delta,abs(v-V[i,j]))
    for i in range(10):
        for j in range(10):
            A = np.zeros(4)
            for k in range(4):
                if (i,j) not in ((0,0),(0,9),(9,0),(9,9)):
                    A[k] = P[k,j*10+i,j*10+max(i-1,0)]*(R[max(i-1,0),j]+r*V[max(i-1,0),j])+\
                        P[k,j*10+i,j*10+min(i+1,9)]*(R[min(i+1,9),j]+r*V[min(i+1,9),j])+\
                        P[k,j*10+i,max(j-1,0)*10+i]*(R[i,max(j-1,0)]+r*V[i,max(j-1,0)])+\
                        P[k,j*10+i,min(j+1,9)*10+i]*(R[i,min(j+1,9)]+r*V[i,min(j+1,9)])
                else:
                    A[k] = P[k,j*10+i,j*10+i]*(R[i,j]+r*V[i,j])
                    if i==0 and j==0: A[k] += P[k,j*10+i,j*10+i+1]*(R[i+1,j]+r*V[i+1,j]) + P[k,j*10+i,(j+1)*10+i]*(R[i,j+1]+r*V[i,j+1])
                    elif i==9 and j==0: A[k] += P[k,j*10+i,j*10+i-1]*(R[i-1,j]+r*V[i-1,j]) + P[k,j*10+i,(j+1)*10+i]*(R[i,j+1]+r*V[i,j+1])
                    elif i==0 and j==9: A[k] += P[k,j*10+i,j*10+i+1]*(R[i+1,j]+r*V[i+1,j]) + P[k,j*10+i,(j-1)*10+i]*(R[i,j-1]+r*V[i,j-1])
                    elif i==9 and j==9: A[k] += P[k,j*10+i,j*10+i-1]*(R[i-1,j]+r*V[i-1,j]) + P[k,j*10+i,(j-1)*10+i]*(R[i,j-1]+r*V[i,j-1])
            Pi[i,j] = np.argmax(A)

    return V, Pi

def plot_map(V, heat_flag):
    plt.figure()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    if heat_flag:
        plt.xticks(np.arange(V.shape[0]))
        plt.yticks(np.arange(V.shape[1]))
        plt.pcolor(V)
        plt.colorbar()
    else:
        plt.xticks(np.arange(V.shape[0]+1))
        plt.yticks(np.arange(V.shape[1]+1))

    for x in np.arange(V.shape[0]):
        for y in np.arange(V.shape[1]):
            plt.gca().text(y+0.5,x+0.5,format(V[x,y], '.1f'),horizontalalignment="center")
    plt.show()

def plot_action(Pi):
    plt.figure()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.xticks(np.arange(Pi.shape[0]+1))
    plt.yticks(np.arange(Pi.shape[1]+1))
    plt.grid(True)
    for x in np.arange(Pi.shape[0]):
        for y in np.arange(Pi.shape[1]):
            if Pi[y,x]==0:
                plt.arrow(x+0.5,y+0.8,0,-0.4,head_width=0.2,head_length=0.2)
            elif Pi[y,x]==1:
                plt.arrow(x+0.5,y+0.2,0,0.4,head_width=0.2,head_length=0.2)
            elif Pi[y,x]==2:
                plt.arrow(x+0.8,y+0.5,-0.4,0,head_width=0.2,head_length=0.2)
            elif Pi[y,x]==3:
                plt.arrow(x+0.2,y+0.5,0.4,0,head_width=0.2,head_length=0.2)
    plt.show()

def extract_reward(lamda, Pi, Rmax, P, n_states, n_actions, r):
    A = set(range(n_actions))
    def T(a,s):
        return np.dot(P[int(Pi[s%10, s//10]),s] - P[a,s], np.linalg.inv(np.eye(n_states) - r*P[int(Pi[s%10, s//10])]))

    c = -np.hstack([np.zeros(n_states), np.ones(n_states),-lamda*np.ones(n_states)])
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))
    T_stack = np.vstack([-T(a,s) for s in range(n_states) for a in A - {int(Pi[s%10, s//10])}])
    I_stack1 = np.vstack([np.eye(1, n_states, s) for s in range(n_states) for a in A - {int(Pi[s%10, s//10])}])
    I_stack2 = np.eye(n_states)
    zero_stack2 = np.zeros((n_states, n_states))

    D_left = np.vstack([T_stack, T_stack, -I_stack2, I_stack2])
    D_middle = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_right = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    b = np.zeros((n_states*(n_actions-1)*2 + 2*n_states, 1))
    #bounds = np.array([(None, None)]*2*n_states + [(-Rmax, Rmax)]*n_states)
    D_bounds = np.hstack([np.vstack([-np.eye(n_states),np.eye(n_states)]), np.vstack([np.zeros((n_states,n_states)), np.zeros((n_states,n_states))]), np.vstack([np.zeros((n_states,n_states)), np.zeros((n_states,n_states))])])
    b_bounds = np.vstack([Rmax*np.ones((n_states,1))]*2)
    D = np.vstack((D, D_bounds))
    b = np.vstack((b, b_bounds))
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    results = solvers.lp(c, A_ub, b)
    r = np.asarray(results["x"][:n_states],dtype=np.double)
    #print(r)
    return np.transpose(r.reshape((10,10)))

def compute_acc(P,R,r,e,Pi_opt):
    V, Pi = compute_pi(P,R,r,e)
    return np.sum(Pi == Pi_opt)/100.0

def plot_acc(lamdas, accs):
    plt.figure()
    plt.plot(lamdas, accs)
    plt.xlabel("lamda")
    plt.ylabel("Accuracy")
    plt.show()
######################
# Main function
######################
def main():
    n_states = 100
    n_actions = 4
    w = 0.1
    r = 0.8
    r_new = 0.2
    e = 0.01
    ### Q1
    R1, R2 = init_R()
    plot_map(R1, True)
    plot_map(R2, True)

    
    P = init_P(w)
    """
    ### Q2
    V1, Pi_1 = compute_pi(P,R1,r,e)
    plot_map(V1, False)
    ### Q3
    plot_map(V1, True)
    ### Q4
    # explain distribution of state values
    ### Q5
    plot_action(Pi_1)
    """
    ### Q6
    V2, Pi_2 = compute_pi(P,R2,r,e)
    plot_map(V2, False)
    ### Q7
    plot_map(V2, True)
    ### Q8
    # explain the distribution of state values
    ### Q9
    plot_action(Pi_2)
    ### Q10

    
    """
    ### Q11
    Rmax_1 = 1
    lamdas = np.linspace(0.0, 5.0, num=500, endpoint=True)
    accs = []
    for lamda in lamdas:
        accs.append(compute_acc(P,extract_reward(lamda, Pi_1, Rmax_1, P, n_states, n_actions, r_new),r_new,e,Pi_1))
    plot_acc(lamdas, accs)

    ### Q12
    best_lamda = lamdas[accs.index(max(accs))]
    print("best lamda: ", best_lamda, " with acc: ", max(accs))

    ### Q13
    R_inv_best = extract_reward(best_lamda, Pi_1, Rmax_1, P, n_states, n_actions, r_new)
    plot_map(R1, 1)
    plot_map(R_inv_best, 1)

    ### Q14
    V_inv_best, Pi_inv_best = compute_pi(P, R_inv_best, r_new, e)
    plot_map(V_inv_best, 1)

    ### Q15

    ### Q16
    plot_action(Pi_inv_best)
    ### Q17
    """
    ### Q18 
    Rmax_2 = 100
    lamdas = np.linspace(0.0, 5.0, num=500, endpoint=True)
    accs = []
    for lamda in lamdas:
        accs.append(compute_acc(P,extract_reward(lamda, Pi_2, Rmax_2, P, n_states, n_actions, r_new),r_new,e,Pi_2))
    plot_acc(lamdas, accs)

    ### Q19
    best_lamda = lamdas[accs.index(max(accs))]
    print("best lamda: ", best_lamda, " with acc: ", max(accs))

    ### Q20
    R_inv_best = extract_reward(best_lamda, Pi_2, Rmax_2, P, n_states, n_actions, r_new)
    plot_map(R2, 1)
    plot_map(R_inv_best, 1)

    ### Q21
    V_inv_best, Pi_inv_best = compute_pi(P, R_inv_best, r_new, e)
    plot_map(V_inv_best, 1)

    ### Q22

    ### Q23
    plot_action(Pi_inv_best)
    ### Q24
    
    ### Q25
    
if __name__ == "__main__":
    main()
