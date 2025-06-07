# %% Imports
import os
from glob import glob
import numpy as np
import mne
import matplotlib.pyplot as plt

# %% Collect all epoched data files
processed_dir = os.path.join('data', 'processed')
epoch_paths = sorted(glob(os.path.join(processed_dir, 'sub-*', '*-epo.fif')))

if not epoch_paths:
    raise FileNotFoundError(f"No epoch files found in {processed_dir}")

# %% Compute power spectra for each subject
psds = []
freqs = None
for path in epoch_paths:
    epochs = mne.read_epochs(path, preload=True, verbose='error')
    # compute power spectral density for each epoch and average across epochs
    psd, freqs = mne.time_frequency.psd_welch(
        epochs, fmin=0.5, fmax=30, n_fft=1024, average='mean', verbose='error'
    )
    # average across channels and epochs
    psds.append(psd.mean(axis=0))

# %% Average power spectra across all subjects
mean_psd = np.mean(psds, axis=0)

# %% Plot the resulting average spectrum
plt.figure()
plt.plot(freqs, mean_psd.T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Average Power Spectrum Across All Subjects')
plt.show()
