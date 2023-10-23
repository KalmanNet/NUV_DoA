import torch
from scipy.signal import find_peaks
import itertools 


# find peaks 1D (known k)
def peak_finding(u, k):
  """
  input: u, tensor of size [m]
         k, number of output peaks
  output: indices of the first k highest peaks
  """
  spectrum = torch.abs(u)
  # Find all peaks and their properties
  peaks, _ = find_peaks(spectrum)

  # If fewer peaks are found than k, raise an error
  if len(peaks) < k:
    raise ValueError('Fewer peaks found than k')

  # Get the heights from the properties dictionary
  peak_heights = spectrum[peaks]

  # Get the indices that would sort the peak heights in descending order
  _, peak_height_indices = torch.topk(peak_heights, k)

  # Get the indices of the first k highest peaks
  peak_indices = peaks[peak_height_indices]

  return peak_indices
  

# Convert to DoA
def convert_to_doa(peak_indices, m):
  """
  input: peak_indices, array of size [k]
         m, number of grids
  output: doa, tensor of size [k]
  """
  doa = peak_indices* (180 / m) - 90
  doa.sort()

  return doa


# MSE computation
def PMSE(pred, DOA):  
    mse_loss = torch.nn.MSELoss()  # Define the MSELoss function
    prmse_list = []

    for p in list(itertools.permutations(pred, len(pred))):
        p = torch.tensor(p)  # Convert each permutation to tensor
        
        # Compute MSE using PyTorch's MSELoss
        prmse_val = mse_loss(p, DOA).item()
        
        prmse_list.append(prmse_val)
        
    return min(prmse_list)