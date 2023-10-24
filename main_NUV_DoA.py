##########################################################
################### simulating NUV-DoA ###################
##########################################################

import torch
import math
import time

from NUV import NUV_DoA

from simulations import utils
from simulations import config
from data.Data_generation import DataGenerator


#### initialization ####
args = config.general_settings()
args.use_cuda = False
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
args.sample = 5
samples_run = args.sample

args.resol = 0.05 # resolution of spatial filter
args.q_init = 0.1 # initial guess of q
centers = torch.linspace(args.B1, args.B2, int((args.B2 - args.B1) / args.resol + 1)).to(device) # centers of all windows
center_num = len(centers) # number of centers
half_window_size = args.resol * 50 # each window covers the interval [center - half_window_size, center + half_window_size]
args.m_SF = int((half_window_size * 2)/args.resol)+1  # number of steering vectors of each SF window, here m_SF=101


#### Generate data ####
generator = DataGenerator(args)
# x_dire, x_true, y_train = generator.generate_experiment_data()
# torch.save([x_dire, x_true, y_train], 'data/DoA_m=360_k=3_sample=5.pt')
[x_dire, x_true, y_train] = torch.load('data/DoA_m=360_k=3_sample=5.pt', map_location=device)

y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

A, A_H = generator.ULA_refine(centers, half_window_size) # generate dictionary matrices for each little window
A = A.to(device)
A_H = A_H.to(device)


# #### estimation ####
import gc
x_pred = torch.zeros(samples_run, center_num, dtype=torch.cfloat, device=device)
theta_tuning = torch.zeros(samples_run, args.k, dtype=torch.float, device=device)
MSE = torch.zeros(samples_run, dtype=torch.float, device=device)

for i in range(samples_run): #r=r_2 * 6000 when k=1
  print(i)
  print('true doa is {}'.format(x_dire[i]))
  r_tuning =  [20000] # [20000, 15000, 10000, 300, 100, 90]
  MSE_tuning = torch.zeros(len(r_tuning), dtype=torch.float, device=device)
  for r in range(len(r_tuning)):
    print('r_tuning = {}'.format(r_tuning[r]))
    # NUV-SSR + SF
    x_pred[i], iterations = NUV_DoA(args, center_num, A, A_H, y_mean[i, :, 0], r_tuning[r])
    # peak finding and convert to DoA
    peak_indices = utils.peak_finding(x_pred[i], args.k)
    theta_tuning[i] = utils.convert_to_doa(peak_indices, centers)
    print('predict doa = {}'.format(theta_tuning[i]))
    # permuted MSE
    MSE_tuning[r] = utils.permuted_mse(theta_tuning[i], x_dire[i])
    print('MSE_tuning = {}'.format(MSE_tuning[r]))
    gc.collect()
    if args.use_cuda: 
        torch.cuda.empty_cache()

  MSE[i] = torch.min(MSE_tuning)
  print('MSE[i] = {}'.format(MSE[i]))
  print('-----------------------------------------')
  gc.collect()
  if args.use_cuda:   
    torch.cuda.empty_cache()

mean_MSE = torch.mean(MSE)
MSE_dB = 10 * (torch.log10(mean_MSE))
print('averaged MSE in dB = {}'.format(MSE_dB))
MSE_linear = torch.sqrt(mean_MSE)
print('averaged RMSE in linear = {}'.format(MSE_linear))
print('--------------------------------------------')
SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {}'.format(SNR))

#### plotting ####
# import matplotlib.pyplot as plt
# import numpy as np
# p = 0

# fig,ax = plt.subplots(figsize=(6, 3))

#   # x_c = xx_pred[p]
#   # axs = axs.ravel()
# x_c = x_pred[p].cpu()
# y_c = np.linspace(args.B1, args.B2, len(x_c), endpoint=False)
#   # y_c = np.linspace(-0.5, 0.5, len(x_c), endpoint=False)
# ax.plot(y_c, x_c, marker = '*')
# # plt.plot(0, x_c[50], "o")
# # ax.set_title('true doa = {}'.format(x_dire[p] ))
#   # argrelextrema(np.array(x_c), np.greater)
# print('true doa = {}'.format(x_dire[p]))
# print('predicted doa = {}'.format(theta[p]))
# print('σ = 3r2 = 30')
