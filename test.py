##########################################################
################### simulating NUV-SSR ###################
##########################################################

import torch
import math
import time

from NUV_SSR import NUV_SSR, NUV_SSR_batched

from simulations import utils
from simulations import config
from Data_generation import DataGenerator


#### initialization ####
args = config.general_settings()
# GPU or CPU
if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")
# number of samples
samples_run = args.sample
# searching the best performed tuning parameter r (std of observation noise)
r_t = [1e-0]

#### Generate data ####
generator = DataGenerator(args)
# # x_dire, x_true, y_train = generator.generate_experiment_data()
# # torch.save([x_dire, x_true, y_train], 'data/Vanilla_m=360_k=3_sample=10.pt')
[x_dire, x_true, y_train] = torch.load('data/Vanilla_m=360_k=3_sample=10.pt', map_location=device)

A = generator.ULA_narrow().to(device) #steering matrix given the true direction x_dire
A_batched = A.reshape(1, args.n, args.m).repeat(samples_run, 1, 1)
A_H_batched = A_batched.transpose(1, 2).conj() 

y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

# #### estimation ####
# Original version
start = time.time()
for r_tuning in r_t:
    print('======================================')
    print('r_tuning = {}'.format(r_tuning))
    x_pred = [0] * samples_run
    theta = [0] * samples_run
    MSE = [0] * samples_run 
    start1 = time.time()
    for i in range(samples_run):
        # NUV-SSR
        [x_pred[i], iterations] = NUV_SSR(args, A, y_mean[i, :, 0], r=r_tuning)
    end1 = time.time()
    t1 = end1 - start1
    # Print Run Time
    print("NUV-SSR origin Run Time:", t1)

    for i in range(samples_run):
        # find peaks
        peak_indices = utils.peak_finding(x_pred[i], args.k)
        # convert to DoA
        theta[i] = utils.convert_to_doa(peak_indices, args.m)  
        # compute MSE
        MSE[i] = utils.PMSE(theta[i], x_dire[i]) # mean square error for each sample     
        print('MSE for {}th sample = {}'.format(i, MSE[i]))
        print('------------------------------------------') 
  
    mean_MSE = sum(MSE) / len(MSE)
    MSE_dB = 10 * (math.log10(mean_MSE))
    print('averaged MSE in dB = {}'.format(MSE_dB))
    MSE_linear = math.sqrt(mean_MSE)
    print('averaged RMSE in linear = {}'.format(MSE_linear))
    print('--------------------------------------------')

end = time.time()
t = end - start
# Print Run Time
print("Total Run Time:", t)

# batched version
start = time.time()
for r_tuning in r_t:
    print('======================================')
    print('r_tuning = {}'.format(r_tuning))
    x_pred = [0] * samples_run
    theta = [0] * samples_run
    MSE = [0] * samples_run 
    start1 = time.time()   
    # NUV-SSR batched
    [x_pred, iterations] = NUV_SSR_batched(args, A_batched, A_H_batched, y_mean, r_tuning)
    end1 = time.time()
    t1 = end1 - start1
    # Print Run Time
    print("NUV-SSR origin Run Time:", t1)

    for i in range(samples_run):
        # find peaks
        peak_indices = utils.peak_finding(x_pred[i], args.k)
        # convert to DoA
        theta[i] = utils.convert_to_doa(peak_indices, args.m)  
        # compute MSE
        MSE[i] = utils.PMSE(theta[i], x_dire[i]) # mean square error for each sample     
        print('MSE for {}th sample = {}'.format(i, MSE[i]))
        print('------------------------------------------') 
  
    mean_MSE = sum(MSE) / len(MSE)
    MSE_dB = 10 * (math.log10(mean_MSE))
    print('averaged MSE in dB = {}'.format(MSE_dB))
    MSE_linear = math.sqrt(mean_MSE)
    print('averaged RMSE in linear = {}'.format(MSE_linear))
    print('--------------------------------------------')

end = time.time()
t = end - start
# Print Run Time
print("Total Run Time:", t)

# SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
# print('SNR = {}'.format(SNR))

#### plotting ####
# import matplotlib.pyplot as plt
# import numpy as np
# # fig,axs = plt.subplots(179)
# p = 0
# fig,ax = plt.subplots(figsize=(5, 3))

# x_c = x_pred[p].cpu()
# y_c = np.linspace(-90, 90, len(x_c), endpoint=False)
#   # y_c = np.linspace(-0.5, 0.5, len(x_c), endpoint=False)
# ax.plot(y_c, x_c, marker = '*')

# # plt.plot(0, x_c[50], "o")
# ax.set_title('M = {}'.format(len(x_c) ))
# plt.savefig('plot/Vanilla_m=360_snr_high.png', format='png')



##########################################################
################### simulating NUV-DoA ###################
##########################################################
#### initialization ####
# args.resol = 0.05 # resolution of spatial filter
# args.q_init = 0.1 # initial guess of q
# centers = torch.linspace(args.B1, args.B2, int((args.B2 - args.B1) / args.resol + 1)).to(device) # centers of all windows
# center_num = len(centers) # number of centers
# half_window_size = args.resol * 50 # each window covers the interval [center - half_window_size, center + half_window_size]
# args.m_SF = int((half_window_size * 2)/args.resol)+1  # number of steering vectors of each SF window, here m_SF=101

# #### Generate data ####
# A, A_H = generator.ULA_refine(centers, half_window_size) # generate dictionary matrices for each little window
# A = A.to(device)
# A_H = A_H.to(device)

# from sklearn.metrics import mean_squared_error
# from itertools import permutations


# # #### estimation ####
# import gc
# x_pred = [0] * samples_run
# MSE = [0] * samples_run
# theta = [0] * samples_run

# for i in range(samples_run): #r=r_2 * 6000 when k=1
#   print(i)
#   print('true doa is {}'.format(x_dire[i]))
#   r_tuning =  [20000, 15000, 10000, 300, 100, 90].to(device)
#   MSE_tuning = [0] * len(r_tuning)
#   for r in range(len(r_tuning)):
#     print('r_tuning = {}'.format(r_tuning[r]))
#     [x_pred[i], theta_tuning, _, iterations, deltas] = estimator.predict(y_mean[i, :, 0], resol = 0.05, k = k, r=r_tuning[r], A = A, A_H = A_H, m = m, max_iterations=3000,  convergence_threshold=1e-3)
#     print('iterations = {}'.format(iterations))
#     print('predicted doa is {}'.format(theta_tuning))
#     MSE_tuning[r] = estimator.PMSE(theta_tuning, x_dire[i])
#     print('MSE_tuning = {}'.format(MSE_tuning[r]))
#     gc.collect()
#     torch.cuda.empty_cache()

#   MSE[i] = min(MSE_tuning)
#   print('MSE[i] = {}'.format(MSE[i]))
#   print('-----------------------------------------')
#   gc.collect()
#   torch.cuda.empty_cache()

# MSE_dB = 10 * (math.log10(np.mean(MSE)))
# print('averaged MSE in dB = {}'.format(MSE_dB))
# MSE_linear = math.sqrt(np.mean(MSE))
# print('averaged RMSE in linear = {}'.format(MSE_linear))
# SNR = 10*math.log10((0.5 + 4) / r2)
# print('SNR = {}'.format(SNR))

# #### plotting ####

# from scipy.signal import argrelextrema
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# p = 0

# fig,ax = plt.subplots(figsize=(6, 3))

#   # x_c = xx_pred[p]
#   # axs = axs.ravel()
# x_c = x_pred[p].cpu()
# y_c = np.linspace(B1, B2, len(x_c), endpoint=False)
#   # y_c = np.linspace(-0.5, 0.5, len(x_c), endpoint=False)
# ax.plot(y_c, x_c, marker = '*')
# # plt.plot(0, x_c[50], "o")
# # ax.set_title('true doa = {}'.format(x_dire[p] ))
#   # argrelextrema(np.array(x_c), np.greater)
# print('true doa = {}'.format(x_dire[p]))
# print('predicted doa = {}'.format(theta[p]))
# print('σ = 3r2 = 30')
