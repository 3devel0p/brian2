#main goal is to simulate oscillations so the model needs to demonstrate that. also maybe model neural noise and refractoriness if needed?
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
runtime = 1*second
time = np.arange(0, int(runtime/defaultclock.dt))*defaultclock.dt
t = np.arange(0, float(runtime/ms), float(defaultclock.dt/ms)) * ms


#oscillations (theta and gamma for hpc, all four for pfc)
theta = 1.0*np.sin(2*np.pi*6*t/second)
gamma = 0.5*np.sin(2*np.pi*40*t/second)
#alpha = 0.4 + 0.4*np.sin(2*np.pi*10*Hz*time)
#beta = 0.4 + 0.4*np.sin(2*np.pi*20*Hz*time)
HPC_oscillations = TimedArray(theta + gamma, dt=defaultclock.dt)
inhibitoryOscillationsHPC = TimedArray(theta, dt=defaultclock.dt)

tau = 10*ms
eqsHPCExcitatory = '''
dv/dt = (HPC_oscillations(t) - v)/tau : 1 (unless refractory)
tau : second
'''
eqsHPCInhibitory = '''
dv/dt = (inhibitoryOscillationsHPC(t) - v)/tau : 1 (unless refractory)
tau : second
'''

excitatory = 100
inhibitory = 25
#threshold is at this value TEMPORARILY
excitatoryHPC = NeuronGroup(excitatory, eqsHPCExcitatory, threshold='v > 1', reset='v = 0', method ='exact')
inhibitoryHPC = NeuronGroup(inhibitory, eqsHPCInhibitory, threshold='v > 1', reset='v = 0', method ='exact')
#the rand method ensures that the neurons aren't all the same. however after the first spike they become identical again. need synapses to fix
excitatoryHPC.v = 'rand()'
inhibitoryHPC.v = 'rand()'

excitatoryHPC.tau = 20*ms
inhibitoryHPC.tau = 10*ms


#to-do: add a normal distrubution for resting, threshold, and reset voltage to simulate noise. crease synapses between excitatory and inhibitory neurons in the hpc and add plasticity. connect synapses based on distance


#synapses (js playing around with it for now, not sure how it will really work)
excitatoryHPCSynapse = Synapses(excitatoryHPC, excitatoryHPC, on_pre='v+=0.2')
excitatoryHPCSynapse.connect(p=0.3)

excitatoryHPCSynapse2 = Synapses(excitatoryHPC, inhibitoryHPC, on_pre='v+=0.2')
excitatoryHPCSynapse2.connect(p=0.3)

inhibitoryHPCSynapse = Synapses(inhibitoryHPC, excitatoryHPC, on_pre='v-=0.4')
inhibitoryHPCSynapse.connect(p=0.3)

inhibitoryHPCSynapse2 = Synapses(inhibitoryHPC, inhibitoryHPC, on_pre='v-=0.4')
inhibitoryHPCSynapse2.connect(p=0.3)
#do the same thing with pfc when added later, but also do one that connects hpc and pfc with a probability of 0.2 and voltage increase of 0.1
#S = Synapses(HPC, HPC, 'w : 1', on_pre='v_post += w')
#S.connect(j='k for k in range(i-3, i+4) if i!=k', skip_if_invalid=True)
#S.w = 'exp(-(x_pre-x_post)**2/(2*width**2))'

#monitors
spikemon_HPC = SpikeMonitor(excitatoryHPC)
statemon_HPC = StateMonitor(excitatoryHPC, 'v', record=True)

M_HPC = PopulationRateMonitor(excitatoryHPC)
run(runtime)

#graphs!!!
plt.figure(figsize=(8,4))
for i in range (100):
    plt.plot(statemon_HPC.t/ms, statemon_HPC.v[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (v)')
plt.title('Voltage Traces')
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(spikemon_HPC.t/ms, spikemon_HPC.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Spike Raster Plot')
plt.show()

'''
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
'''
