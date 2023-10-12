import numpy as np
from scipy.signal import find_peaks

# find peaks (known k)
def peak_finding(u, k, m):
  Spectrum = abs(u)
  x = np.linspace(0, m, m, endpoint=False)
  DOA_pred,_ = find_peaks(Spectrum)
  DOA_pred = list(DOA_pred)
  DOA_pred.sort(key = lambda x: Spectrum[x], reverse = True)
  DOA = []
  for i in DOA_pred[0:k]:
    DOA.append(i* (180 / m) - 90)
  DOA.sort()

  return DOA