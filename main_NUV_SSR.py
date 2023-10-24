##########################################################
################### simulating NUV-SSR ###################
##########################################################

import torch
import math
import time

from NUV import NUV_SSR, NUV_SSR_batched

from simulations import utils
from simulations import config
from data.Data_generation import DataGenerator


#### initialization ####
args = config.general_settings()
args.use_cuda = True
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
args.sample = 100
samples_run = args.sample
# searching the best performed tuning parameter r (std of observation noise)
r_t = [1e-0]

#### Generate data ####
generator = DataGenerator(args)
# # x_dire, x_true, y_train = generator.generate_experiment_data()
# # torch.save([x_dire, x_true, y_train], 'data/Vanilla_m=360_k=3_sample=100.pt')
[x_dire, x_true, y_train] = torch.load('data/Vanilla_m=360_k=3_sample=100.pt', map_location=device)

A = generator.ULA_narrow().to(device) #steering matrix given the true direction x_dire
A_batched = A.reshape(1, args.n, args.m).repeat(samples_run, 1, 1)
A_H_batched = A_batched.transpose(1, 2).conj() 

y_mean = y_train.mean(dim=1) # generate y_mean by averaging l snapshots for each sample

# #### estimation ####
# batched version
start = time.time()
for r_tuning in r_t:
    print('======================================')
    print('r_tuning = {}'.format(r_tuning))
    if args.use_cuda:       
        # NUV-SSR batched
        x_pred, iterations = NUV_SSR_batched(args, A_batched, A_H_batched, y_mean, r_tuning)
    else:
        # NUV-SSR using for loop
        x_pred = torch.zeros(samples_run, args.m, dtype=torch.cfloat, device=device)
        iterations = torch.zeros(samples_run, dtype=torch.int, device=device)
        for i in range(samples_run):
            x_pred[i], iterations[i] = NUV_SSR(args, A, y_mean[i], r_tuning)

    # find peaks
    peak_indices = utils.batch_peak_finding(x_pred, args.k)
    # convert to DoA
    theta = utils.batch_convert_to_doa(peak_indices, args.m)  
    # compute MSE
    MSE = utils.batched_permuted_mse(theta, x_dire) # mean square error for all samples   
  
    mean_MSE = torch.mean(MSE)
    MSE_dB = 10 * (torch.log10(mean_MSE))
    print('averaged MSE in dB = {}'.format(MSE_dB))
    MSE_linear = torch.sqrt(mean_MSE)
    print('averaged RMSE in linear = {}'.format(MSE_linear))
    print('--------------------------------------------')

end = time.time()
t = end - start
# Print Run Time
print("Total Run Time:", t)
SNR = 10*math.log10((args.x_var + args.mean_c) / args.r2)
print('SNR = {}'.format(SNR))

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