import torch
import numpy as np


def NUV_SSR(ss_model, y, r, max_iterations, convergence_threshold):
    """
    1. quantize the azimuth into m equidistant grid cells
       so each cell length is theta/m

    2. apply EM to estimate to posteriori distribution of decision vector q (size m)
       i.e. mean and variance
    """
   
    A = ss_model.A
    A_H = ss_model.A_H
    m = ss_model.m
    n = ss_model.n
    l = ss_model.l

    delta = [] # delta records the difference of norm of q between each iteration it+1 and it

    ### 1. Initial Guess ###
    q = 0.01 * torch.ones(max_iterations + 2, m, dtype=torch.cdouble).cpu() 


    ### 2. EM Algorithm ###
    iterations = max_iterations

    for it in range(max_iterations):
        # 2a. Precision Matrix
        q[it] = q[it]

        torch.diag(torch.square(q[it]))

        W_inv = torch.zeros(n, n, dtype=torch.cdouble)
        W_inv =  A @ torch.diag(torch.square(q[it])) @ A_H 

        W_inv = W_inv + (r/l) * torch.eye(ss_model.n, dtype = torch.cdouble)          
        W = torch.zeros(n, n, dtype=torch.cdouble)
        W = torch.linalg.inv(W_inv)

        # 2b. Gaussian Posteriori Distribution
        mean = torch.zeros(m, dtype=torch.cdouble) 
        variance = torch.zeros(m, dtype=torch.cdouble)

        mean = abs(torch.diag(torch.square(q[it])) @ A_H @ W @ y )
        variance =  abs(torch.pow(q[it], 2) -  torch.diag(torch.pow(q[it], 4)) @ torch.diagonal(A_H @ W @ A) )
        q[it+1] = torch.sqrt(torch.square(mean) + variance)
        
        if torch.norm(q[it + 1] - q[it]) < convergence_threshold:   # stopping criteria
            q[-1] = q[it + 1]
            iterations = it
            break
        else:
            delta.append(torch.norm(q[it + 1] - q[it]))
        

        q[-1]=q[it+1]


    ### 3. MAP Estimator of the sparse signal###

    u = torch.zeros(l, ss_model.m, dtype = torch.cdouble)
    u = torch.diag(torch.square(q[-1])) @ A_H @ W @ y

    
    return [abs(q[-1]), abs(u), iterations, delta]


