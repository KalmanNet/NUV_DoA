import torch

def NUV_SSR(args, A, y, r):
   """
   1. quantize the signal range into m equidistant grid cells
      so each cell length is theta/m

   2. apply EM to estimate to posteriori distribution of decision vector q (size m)
      i.e. mean and variance
   """
   # Set up parameters
   A_H = A.conj().T
   m = args.m
   n = args.n
   l = args.l
   max_iterations = args.max_iterations
   convergence_threshold = args.convergence_threshold

   ### 1. Initial Guess ###
   q = args.q_init * torch.ones(2, m, dtype=torch.cfloat, device=A.device)

   ### 2. EM Algorithm ###
   iterations = max_iterations  

   for it in range(max_iterations):
      # 2a. Precision Matrix
      q[0] = q[1]
   
      W_inv =  A @ torch.diag(torch.square(q[0])) @ A_H 
      W_inv = W_inv + (r/l) * torch.eye(n, dtype = torch.cfloat, device = A.device)          
      
      W = torch.linalg.inv(W_inv)

      # 2b. Gaussian Posteriori Distribution
      mean = torch.abs(torch.diag(torch.square(q[0])) @ A_H @ W @ y )
      variance =  torch.abs(torch.pow(q[0], 2) -  torch.diag(torch.pow(q[0], 4)) @ torch.diagonal(A_H @ W @ A) )
      q[1] = torch.sqrt(torch.square(mean) + variance)
      
      if torch.norm(q[1] - q[0]) < convergence_threshold:   # stopping criteria
         iterations = it
         break
   

   ### 3. MAP Estimator of the sparse signal###
   u = torch.diag(torch.square(q[1])) @ A_H @ W @ y

   return [u, iterations]


def NUV_SSR_batched(args, A, A_H, y, r):   
   # Set up parameters
   m = args.m
   n = args.n
   l = args.l
   max_iterations = args.max_iterations
   convergence_threshold = args.convergence_threshold

   samples_run = args.sample

   id = torch.eye(n, dtype=torch.cfloat, device=A.device)
   id = id.reshape((1, n, n))
   id_batch = id.repeat(samples_run, 1, 1)

   # 1. Initial Guess
   q = args.q_init * torch.ones(samples_run, 2, m, dtype=torch.cfloat, device=A.device)

   # 2. EM Algorithm
   iterations = max_iterations

   for it in range(max_iterations):
      q[:, 0, :] = q[:, 1, :]
      
      diag_squared_q = torch.diag_embed(torch.square(q[:, 0, :]), offset=0, dim1=-2, dim2=-1)
      W_inv = A @ diag_squared_q @ A_H
      W_inv = W_inv + (r / l) * id_batch

      W = torch.linalg.inv(W_inv)

      mean = torch.abs(diag_squared_q @ A_H @ W @ y)

      M1 = torch.diag_embed(torch.pow(q[:, 0, :], 4), offset=0, dim1=-2, dim2=-1)
      M2 = torch.diagonal(A_H @ W @ A, dim1=-2, dim2=-1)
      M2 = M2.unsqueeze(-1)
      
      variance = torch.abs(torch.pow(q[:, 0, :], 2) - torch.squeeze(torch.bmm(M1, M2),2))

      q[:, 1, :] = torch.sqrt(torch.squeeze(torch.square(mean),2) + variance)

      if max(torch.linalg.norm(q[:, 0, :] - q[:, 1, :], dim=1)) < convergence_threshold:
            iterations = it
            break

   # 3. MAP Estimator
   U = torch.diag_embed(torch.square(q[:, -1, :]), offset=0, dim1=-2, dim2=-1) @ A_H @ W @ y
   U = torch.squeeze(U,2)
   
   return [U, iterations]


