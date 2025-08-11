!pip install brian2 numpy scipy matplotlib statsmodels scikit-learn
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
from sklearn.metrics import mutual_info_score

start_scope()
defaultclock.dt = 0.1*ms
runtime = 500*ms
time = np.arange(0, int(runtime/defaultclock.dt))*defaultclock.dt

tau = 10*ms
eqs = '''
dv/dt = (I - v)/tau : 1 (unless refractory)
I : 1
'''

n = 100
HPC = NeuronGroup(n, eqs, threshold='v > 1', reset='v = 0', method ='exact')
PFC = NeuronGroup(n, eqs, threshold='v > 1', reset='v = 0', method ='exact')
HPC.v = 'rand()'
PFC.v = 'rand()'

theta = 0.5 + 0.5*np.sin(2*np.pi*6*Hz*time)
gamma = 0.3 + 0.3*np.sin(2*np.pi*40*Hz*time)
alpha = 0.4 + 0.4*np.sin(2*np.pi*10*Hz*time)
beta = 0.4 + 0.4*np.sin(2*np.pi*20*Hz*time)

hpc_input = theta + gamma
pfc_input = alpha + beta

hpc_drive = TimedArray(hpc_input, dt=defaultclock.dt)
pfc_drive = TimedArray(pfc_input, dt=defaultclock.dt)

HPC.run_regularly('I = hpc_drive(t)', dt=defaultclock.dt)
PFC.run_regularly('I = pfc_drive(t)', dt=defaultclock.dt)

spikemon_HPC = SpikeMonitor(HPC)
statemon_HPC = StateMonitor(HPC, 'v', record=True)
spikemon_PFC = SpikeMonitor(PFC)
statemon_PFC = StateMonitor(PFC, 'v', record=True)
M_HPC = PopulationRateMonitor(HPC)
M_PFC = PopulationRateMonitor(PFC)
run(runtime)

v_data = statemon_HPC.v / volt  # Corrected unit
vdata2 = statemon_PFC.v / volt  # Corrected unit

rate_HPC = M_HPC.smooth_rate(window='flat', width=5*ms)[::10] / Hz
rate_PFC = M_PFC.smooth_rate(window='flat', width=5*ms)[::10] / Hz
fs = 1 / (defaultclock.dt/second) / 10

fft_hpc = fft(rate_HPC)
fft_pfc = fft(rate_PFC)
freqs = fftfreq(len(rate_HPC), 1/fs)

plt.figure(figsize=(10,4))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_hpc[:len(freqs)//2]), label='HPC')
plt.plot(freqs[:len(freqs)//2], np.abs(fft_pfc[:len(freqs)//2]), label='PFC')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT: HPC vs. PFC')
plt.legend()
plt.show()

analytic_hpc = hilbert(rate_HPC)
analytic_pfc = hilbert(rate_PFC)
phase_hpc = np.angle(analytic_hpc)
phase_pfc = np.angle(analytic_pfc)

plv = np.abs(np.mean(np.exp(1j * (phase_hpc - phase_pfc))))
print(f'PLV (HPC vs PFC): {plv:.3f}')

data = pd.DataFrame({
    'HPC': rate_HPC,
    'PFC': rate_PFC
})

print('Granger Causality: HPC -> PFC')
grangercausalitytests(data[['PFC', 'HPC']], maxlag=10, verbose=True)
print('\nGranger Causality: PFC -> HPC')
grangercausalitytests(data[['HPC', 'PFC']], maxlag=10, verbose=True)

bins = 20
hpc_binned = np.digitize(rate_HPC, np.histogram(rate_HPC, bins=bins)[1])
pfc_binned = np.digitize(rate_PFC, np.histogram(rate_PFC, bins=bins)[1])

mi = mutual_info_score(hpc_binned, pfc_binned)
print(f'Mutual Information (HPC - PFC): {mi:.3f} bits')

#raster plot
plt.figure(figsize=(12,4))
plt.plot(spikemon_HPC.t/ms, spikemon_HPC.i, '.k', label="HPC")
plt.plot(spikemon_PFC.t/ms, spikemon_PFC.i, '.r', label="PFC")
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Raster Plot: HPC (black) + PFC (red)')
plt.legend()
plt.show()

#spectrogram
rate_HPC = M_HPC.smooth_rate(window='flat', width=5*ms)/Hz
rate_PFC = M_PFC.smooth_rate(window='flat', width=5*ms)/Hz

f_HPC, t_HPC, Sxx_HPC = spectrogram(rate_HPC, fs=1/(defaultclock.dt/second))
f_PFC, t_PFC, Sxx_PFC = spectrogram(rate_PFC, fs=1/(defaultclock.dt/second))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.pcolormesh(t_HPC*1000, f_HPC, Sxx_HPC, shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (ms)')
plt.title('HPC Spectrogram')
plt.ylim(0, 100)

plt.subplot(1,2,2)
plt.pcolormesh(t_PFC*1000, f_PFC, Sxx_PFC, shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (ms)')
plt.title('PFC Spectrogram')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

#power spectrum
f_HPC, Pxx_HPC = welch(rate_HPC, fs=1/(defaultclock.dt/second), nperseg=1024)
f_PFC, Pxx_PFC = welch(rate_PFC, fs=1/(defaultclock.dt/second), nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(f_HPC, Pxx_HPC, label='HPC')
plt.semilogy(f_PFC, Pxx_PFC, label='PFC')
plt.title('Power Spectrum (HPC & PFC')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.grid(True)
plt.legend()
plt.show()

#heatmap
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.imshow(statemon_HPC.v/mV, aspect='auto', cmap='hot', extent=[statemon_HPC.t[0]/ms, statemon_HPC.t[-1]/ms, 0, n])
plt.colorbar(label='V (mV)')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron (HPC)')
plt.title('HPC Heatmap')

plt.subplot(1,2,2)
plt.imshow(statemon_PFC.v/mV, aspect='auto', cmap='hot', extent=[statemon_PFC.t[0]/ms, statemon_PFC.t[-1]/ms, 0, n])
plt.colorbar(label='V (mV)')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron (PFC)')
plt.title('PFC Heatmap')

plt.tight_layout()
plt.show()
