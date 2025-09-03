%matplotlib inline
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

#CONTROL DATA NEURON STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
duration = 150*ms
taupre = taupost = 20*ms
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05
wmax = 1.0
tmax = 50*ms

v0_max = 3
sigma = 0.2
eqs = '''
dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
v0 : 1
tau : second
'''

hpc = NeuronGroup(100, eqs, threshold='v>0.8', reset='v = 0', refractory=15*ms, method='euler')
hpc.v = 'rand()'
hpc.tau = 10*ms

S = Synapses(hpc, hpc, '''
             w : 1
             dapre/dt = -apre/taupre : 1 (clock-driven)
             dapost/dt = -apost/taupost : 1 (clock-driven)
             ''', 
             on_pre='''
             v_post += w
             apre += Apre
             w = clip(w+apost, 0, wmax)
             ''',
             on_post='''
             apost += Apost
             w = clip(w+apre, 0, wmax)
             ''', method='linear')
S.connect(condition="i!=j", p=0.2)
S.w = 'j*0.2'
S.delay = 'j*2*ms'


M = StateMonitor(hpc, 'v', record=True)
spikemon = SpikeMonitor(hpc)
pop = PopulationRateMonitor(hpc)

hpc.v0 = 'i*v0_max/(N-1)'

run(duration)

#graphs and data collection!!!!
fs = 1 / (defaultclock.dt/second) / 10
rate = pop.smooth_rate(window='flat', width=5*ms)/Hz
#rate_PFC = M_PFC.smooth_rate(window='flat', width=5*ms)/Hz

fft1 = fft(rate)
#fft_pfc = fft(rate_PFC)
freqs = fftfreq(len(rate), 1/fs)

#synaptic connections scatterplot
plt.figure(figsize=(8,4))
plt.scatter(S.i, S.j, s=S.w*10, c=S.w, cmap='viridis')
plt.xlabel('Pre-synaptic Neuron Index')
plt.ylabel('Post-synaptic Neuron Index')
plt.title('Synaptic Connections')
plt.colorbar(label='Synaptic Weight')
plt.show()

#voltage traces
plt.figure(figsize=(8,4))
for i in range (100):
    plt.plot(M.t/ms, M.v[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (v)')
plt.title('Voltage Traces')
plt.legend()
plt.show()

#fft
plt.figure(figsize=(10,4))
plt.plot(freqs[:len(freqs)//2], np.abs(fft1[:len(freqs)//2]), label='group 1')
#plt.plot(freqs[:len(freqs)//2], np.abs(fft_pfc[:len(freqs)//2]), label='PFC')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT group 1')
#plt.legend()
plt.show()

#plv (come back to this later, figure out how to do it within instead of between groups)
analytic1 = hilbert(rate)
#analytic_pfc = hilbert(rate_PFC)
#phase1 = np.angle(analytic1)
#phase_pfc = np.angle(analytic_pfc)

#plv = np.abs(np.mean(np.exp(1j * (phase_hpc - phase_pfc))))
#print(f'PLV (HPC vs PFC): {plv:.3f}')

#raster plot
plt.figure(figsize=(12,4))
plt.plot(spikemon.t/ms, spikemon.i, '.k', label="group 1")
#plt.plot(spikemon_PFC.t/ms, spikemon_PFC.i, '.r', label="PFC")
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Raster Plot: HPC (black) + PFC (red)')
#plt.legend()
plt.show()

#spectrogram

f1, t1, Sxx1 = spectrogram(rate, fs=1/(defaultclock.dt/second))
#f_PFC, t_PFC, Sxx_PFC = spectrogram(rate_PFC, fs=1/(defaultclock.dt/second))

plt.figure(figsize=(12,4))
#plt.subplot(1,2,1)
plt.pcolormesh(t1*1000, f1, Sxx1, shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (ms)')
plt.title('HPC Spectrogram')
plt.ylim(0, 100)

#plt.subplot(1,2,2)
#plt.pcolormesh(t_PFC*1000, f_PFC, Sxx_PFC, shading='gouraud')
#plt.ylabel('Frequency (Hz)')
#plt.xlabel('Time (ms)')
#plt.title('PFC Spectrogram')
#plt.ylim(0, 100)

plt.tight_layout()
plt.show()

#heatmap (figure out how to do this properly as well)

#power spectrum
f, Pxx = welch(rate, fs=1/(defaultclock.dt/second), nperseg=1024)
#f_PFC, Pxx_PFC = welch(rate_PFC, fs=1/(defaultclock.dt/second), nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(f, Pxx, label='HPC')
#plt.semilogy(f_PFC, Pxx_PFC, label='PFC')
plt.title('Power Spectrum (group 1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.grid(True)
plt.legend()
plt.show()

#granger causality (same as plv)
'''data = pd.DataFrame({
    'HPC': rate_HPC,
    'PFC': rate_PFC
})'''

print('Granger Causality: HPC -> PFC')
#grangercausalitytests(data[['PFC', 'HPC']], maxlag=10, verbose=True)
print('\nGranger Causality: PFC -> HPC')
#grangercausalitytests(data[['HPC', 'PFC']], maxlag=10, verbose=True)

#mutual information (same as plv)
bins = 20
#hpc_binned = np.digitize(rate_HPC, np.histogram(rate_HPC, bins=bins)[1])
#pfc_binned = np.digitize(rate_PFC, np.histogram(rate_PFC, bins=bins)[1])

#mi = mutual_info_score(hpc_binned, pfc_binned)
#print(f'Mutual Information (HPC - PFC): {mi:.3f} bits')

#cognitive tasks

#AD GROUP STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#OPEN LOOP GROUP STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#mutual info
