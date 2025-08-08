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
duration = 1*second
time = np.arange(0, int(duration/defaultclock.dt))*defaultclock.dt

eqs = '''
dv/dt = (I - v)/(10*ms) : 1
I : 1
'''
N_HPC = 100
N_PFC = 100
HPC = NeuronGroup(N_HPC, eqs, threshold='v > 1', reset='v = 0', method ='exact')
PFC = NeuronGroup(N_PFC, eqs, threshold='v > 1', reset='v = 0', method ='exact')

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
run(duration)

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
