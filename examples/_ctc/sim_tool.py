#!/usr/bin/env python

import matplotlib.pyplot as plt
import random as rand
import numpy as np

# ground-truth label sequence, NOTE that 0 indicates blank
l = [0,2,0,1,0,3,0]

U = len(l)          # length of label sequence
C = max(l) + 1      # number of classes
T = 20              # time steps, TRY a layger T for example 50 to see the difference
lr = 0.1            # learning rate

colors = "bgrcmykw"

def ctc(pred):
    ctc_alpha = np.zeros((U,T))
    ctc_alpha[0,0] = pred[l[0],0]
    if l[0] == 0:
        ctc_alpha[1,0] = pred[l[1],0]
    for t in range(1,T):
        for u in range(U):
            pr = ctc_alpha[u,t-1]
            if u > 0:
                pr += ctc_alpha[u-1,t-1]
                if u > 1 and l[u-1] == 0 and l[u-2] != l[u]:
                    pr += ctc_alpha[u-2,t-1]
            ctc_alpha[u,t] = pr * pred[l[u],t]
            
    p_seq = ctc_alpha[U-1,T-1]
    ctc_beta = np.zeros((U,T))
    ctc_beta[U-1,T-1] = 1
    if l[-1] == 0:
        ctc_beta[U-2,T-1] = 1
        p_seq +=  ctc_alpha[U-2,T-1]
    for t in range(T-2,-1,-1):
        for u in range(U):
            ctc_beta[u,t] = ctc_beta[u,t+1] * pred[l[u],t+1]
            if u < U-1:
                ctc_beta[u,t] += ctc_beta[u+1,t+1] * pred[l[u+1],t+1]
                if u < U-2 and l[u+1] == 0 and l[u+2] != l[u]:
                    ctc_beta[u,t] += ctc_beta[u+2,t+1] * pred[l[u+2],t+1]

    targ = np.zeros((C,T))
    for t in range(T):
        for u in range(U):
            targ[l[u],t] += ctc_alpha[u,t] * ctc_beta[u,t]
    targ /= p_seq

    return targ, p_seq

data = np.random.rand(C,T) * 0.01 # randomly initialization

# the modification on CTC training, refer to "Reinterpreting CTC training as iterative fitting" for more details
alpha = 0           # used to set non-blank proportion
gamma = 0           # focusing on key frames

for iter in range(100000000):
    pred = np.exp(data)
    pred /= np.tile(np.sum(pred, axis=0), (C,1))
    targ, p_seq = ctc(pred)
    print('iteration',iter,', probability of correct labelling =',p_seq)
    
    if alpha:
        ca = np.sum(targ, axis=1)
        sum_c = 0
        for c in range(1,C):
            sum_c += np.sum(l.count(c))
            targ[c,:] *= np.sum(l.count(c)) / max(0.00000001, ca[c])
        targ[0,:] *= (1 - alpha) * sum_c / (max(0.00000001, ca[0]) * alpha)
        ps = np.sum(targ, axis=0)
        targ /= np.tile(ps, (C,1))

    grad = pred - targ

    if gamma:
        weight = np.max(targ - pred, axis=0)
        weight = np.power(weight, gamma)
        weight *= T / np.sum(weight)
        grad *= np.tile(weight, (C,1))
    
    data -= grad * lr
    
    if iter % 10 == 0:  # display
        plt.ylim(-0.1,1.1)
        plt.xticks([])
        for c in range(1,C):
            plt.plot([t for t in range(T)], list(pred[c,:]),'-',color=colors[c%8])
        for c in range(1,C):
            plt.plot([t for t in range(T)], list(targ[c,:]), '--',color=colors[c%8])
        plt.show()