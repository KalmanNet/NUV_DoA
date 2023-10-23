import torch
import itertools 


# find peaks 1D (known k), suport GPU
def batch_peak_finding(u, k):
    """
    input: u, tensor of size [batch size, m]
           k, number of output peaks
    output: indices of the first k highest peaks [batch size, k]
    """
    spectrum = torch.abs(u)

    # Take the difference to find where the derivative changes sign
    diff = spectrum[:, 1:] - spectrum[:, :-1]
    
    # Find peaks (where the difference changes from positive to negative)
    peaks = (diff[:, :-1] > 0) & (diff[:, 1:] < 0)

    # Check if any batch has fewer than k peaks
    peaks_count = peaks.sum(dim=1)
    if torch.any(peaks_count < k):
        batch_ids = torch.nonzero(peaks_count < k, as_tuple=True)[0]
        raise ValueError(f'Fewer peaks found than k for batches: {batch_ids.tolist()}')

    # Adjust spectrum to match the shape of peaks for the subsequent masking
    spectrum_adjusted = spectrum[:, 1:-1]

    # Mask out the non-peak values with very negative numbers so they don't interfere with topk
    masked_spectrum = torch.where(peaks, spectrum_adjusted, torch.tensor(float('-inf'), device=u.device))

    # Find the top k peaks for each batch; values give the heights, indices give the locations
    values, batched_peak_indices = masked_spectrum.topk(k, dim=1)

    # Adjust indices to account for the shifted spectrum
    batched_peak_indices += 1

    return batched_peak_indices
  

# Convert to DoA
def batch_convert_to_doa(peak_indices, m):
  """
  input: peak_indices, tensor of size [batch_size, k]
          m, number of grids
  output: doa, tensor of size [batch_size, k]
  """
  # Convert peak indices to doa
  doa = peak_indices * (180 / m) - 90

  # Sort each batch
  doa, _ = torch.sort(doa, dim=1)

  return doa


# permuted MSE computation
def all_permutations(length):
    """Return all permutations of a sequence of given length."""
    return torch.tensor(list(itertools.permutations(range(length))), dtype=torch.long)

def batched_permuted_mse(pred, DOA):
    batch_size, k = pred.shape
    device = pred.device
    
    # Step 1: Generate all possible permutations of indices [0, 1, ..., k-1]
    perms = all_permutations(k).to(device)  # [k!, k]
    # Expand perms to match batch_size
    perms = perms.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, k!, k]
    
    # Step 2: Use these permutations to reorder each sample in the batch
    # Expand dims for broadcasting with permutations
    expanded_pred = pred.unsqueeze(1)  # [batch_size, 1, k]
    expanded_pred = expanded_pred.repeat(1, perms.shape[1], 1)  # [batch_size, k!, k]
    
    # Gather results according to permutations
    permuted_preds = torch.gather(expanded_pred, 2, perms)  # [batch_size, k!, k]

    # Step 3: Compute the MSE for each permutation of each sample
    mse = torch.mean((permuted_preds - DOA.unsqueeze(1))**2, dim=-1)  # [batch_size, k!]

    # Step 4: Find the minimum MSE for each sample
    min_mse, _ = torch.min(mse, dim=-1)  # [batch_size]

    return min_mse
