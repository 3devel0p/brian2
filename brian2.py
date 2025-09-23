# checklist for hpc-pfc model
# 1. define hpc and pfc neuron groups with different membrane constants, etc.
# 2. define synaptic connections both between and within groups
# 3. traits that are different between hpc and pfc neurons: excitability/threshold, plasticity/synaptic weights, learning rate (taupre/taupost), inputs and outputs (represented by v0), and oscillatory activity (tau) 
%matplotlib inline
!pip install brian2 numpy scipy matplotlib statsmodels scikit-learn
#function definitions and imports
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, filtfilt, butter
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
from sklearn.metrics import mutual_info_score

def singleRegionPLV(spikebin):
    all_phases = np.zeros((100, time_steps))

    for i in range(100):
        filtered_signal = filtfilt(b, a, spikebin[i, :])
        analytic_signal = hilbert(filtered_signal)
        instantaneous_phase = np.angle(analytic_signal)
        all_phases[i, :] = instantaneous_phase

    complex_phases = np.exp(1j * all_phases)
    mean_vector = np.mean(complex_phases, axis=0)
    plv_over_time = np.abs(mean_vector)
    average_plv = np.mean(plv_over_time)
    return average_plv

def singleRegionMI(spikebin):
    mi_matrix = np.zeros((100, 100))

    for i in range(100):
        for j in range(i, 100): 
            if i == j:
                hist, _ = np.histogram(spikebin[i, :], bins=[-0.5, 0.5, 1.5])
                p_hist = hist / np.sum(hist)
                entropy = -np.sum(p_hist * np.log2(p_hist + epsilon))
                mi_matrix[i, j] = entropy
            else:
                mi = mutual_info_score(labels_true=spikebin[i, :], labels_pred=spikebin[j, :])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi 

    return np.around(mi_matrix, 3)

#CONTROL DATA NEURON STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
duration = 150*ms
taupre = taupost = 20*ms
#pfc has lower learning rate?
taupre_pfc = taupost_pfc = 10*ms
Apre = 0.01
epsilon = 1e-10 #come back to this later
Apost = -Apre*taupre/taupost*1.05
wmax = 1.0
tmax = 50*ms
freq_mod = 10 * Hz #come back to this frequency of modulating input later

#different v0 values represent inputs and outputs (find citation for this)
v0_max = 3
v0_max_pfc = 1.5
sigma = 0.2
eqs = '''
dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
v0 : 1
tau : second
'''

#hpc has lower threshold
hpc = NeuronGroup(100, eqs, threshold='v>0.7', reset='v = 0', refractory=15*ms, method='euler')
pfc = NeuronGroup(100, eqs, threshold='v>0.8', reset='v = 0', refractory=15*ms, method='euler')
hpc.v = 'rand()'
hpc.tau = 10*ms
pfc.v = 'rand()'
pfc.tau = 20*ms

#synapses (maybe add inhibitory neurons to inhibit surrounding excitatory neurons. your data will still be about excitatory neurons)

#hpc has stronger connections so higher synaptic weights while pfc has slower learning so lower taupre/taupost i think (find citation)
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

S2 = Synapses(hpc, pfc, '''
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
S2.connect(condition="i!=j", p=0.1)
S2.w = 'j*0.2'
S2.delay = 'j*2*ms'

S3 = Synapses(pfc, pfc, '''
             w : 1
             dapre/dt = -apre/taupre_pfc : 1 (clock-driven)
             dapost/dt = -apost/taupost_pfc : 1 (clock-driven)
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
S3.connect(condition="i!=j", p=0.2)
S3.w = 'j*0.1'
S3.delay = 'j*2*ms'

#pfc to hpc connections are sparse (find citation for this) and figure out whether to use taupre or taupre_pfc for this
S4 = Synapses(pfc, hpc, '''
             w : 1
             dapre/dt = -apre/taupre_pfc : 1 (clock-driven)
             dapost/dt = -apost/taupost_pfc : 1 (clock-driven)
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
S4.connect(condition="i!=j", p=0.02)
S4.w = 'j*0.2'
S4.delay = 'j*2*ms'

M = StateMonitor(hpc, 'v', record=True)
M_PFC = StateMonitor(pfc, 'v', record=True)
spikemon = SpikeMonitor(hpc)
spikemon_pfc = SpikeMonitor(pfc)
pop = PopulationRateMonitor(hpc)
pop_pfc = PopulationRateMonitor(pfc)

hpc.v0 = 'i*v0_max/(N-1)'
pfc.v0 = 'i*v0_max_pfc/(N-1)'

run(duration)

#graphs and data collection!!!!
fs = 1 / (defaultclock.dt/second) / 10
rate = pop.smooth_rate(window='flat', width=5*ms)/Hz
rate_PFC = pop_pfc.smooth_rate(window='flat', width=5*ms)/Hz

fft1 = fft(rate)
fft_pfc = fft(rate_PFC)
freqs = fftfreq(len(rate), 1/fs)

#synaptic connections scatterplot that includes all four synapse groups
plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.scatter(S.i, S.j, s=S.w*10, c=S.w, cmap='viridis')
plt.xlabel('Pre-synaptic Neuron Index')
plt.ylabel('Post-synaptic Neuron Index')
plt.title('Synaptic Connections (HPC -> HPC)')

plt.subplot(2,2,2)
plt.scatter(S2.i, S2.j, s=S2.w*10, c=S2.w, cmap='viridis')
plt.xlabel('Pre-synaptic Neuron Index')
plt.ylabel('Post-synaptic Neuron Index')
plt.title('Synaptic Connections (HPC -> PFC)')

plt.subplot(2,2,3)
plt.scatter(S3.i, S3.j, s=S3.w*10, c=S3.w, cmap='viridis')
plt.xlabel('Pre-synaptic Neuron Index')
plt.ylabel('Post-synaptic Neuron Index')
plt.title('Synaptic Connections (PFC -> PFC)')

plt.subplot(2,2,4)
plt.scatter(S4.i, S4.j, s=S4.w*10, c=S4.w, cmap='viridis')
plt.xlabel('Pre-synaptic Neuron Index')
plt.ylabel('Post-synaptic Neuron Index')
plt.title('Synaptic Connections (PFC -> HPC)')

plt.colorbar(label='Synaptic Weight')
plt.tight_layout()
plt.show()

#voltage traces (hpc)
plt.figure(figsize=(8,4))
for i in range (100):
    plt.plot(M.t/ms, M.v[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (v)')
plt.title('Voltage Traces (HPC)')
plt.legend()
plt.show()

#voltage traces (pfc)
plt.figure(figsize=(8,4))
for i in range (100):
    plt.plot(M_PFC.t/ms, M_PFC.v[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (v)')
plt.title('Voltage Traces (PFC)')
plt.legend()
plt.show()

#fft
plt.figure(figsize=(10,4))
plt.plot(freqs[:len(freqs)//2], np.abs(fft1[:len(freqs)//2]), label='HPC')
plt.plot(freqs[:len(freqs)//2], np.abs(fft_pfc[:len(freqs)//2]), label='PFC')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT (HPC and PFC)')
plt.legend()
plt.show()

#plv (definitely need to make sure this is correct)
# ------ hpc to pfc
analytic1 = hilbert(rate)
analytic_pfc = hilbert(rate_PFC)
phase_hpc = np.angle(analytic1)
phase_pfc = np.angle(analytic_pfc)

plv = np.abs(np.mean(np.exp(1j * (phase_hpc - phase_pfc))))
print(f'PLV (HPC vs PFC): {plv:.3f}')

# ------ hpc to hpc
time_steps = int(duration / defaultclock.dt)

binned_spikes = np.zeros((100, time_steps))

for i in range(100):
    neuron_spikes = spikemon.t[spikemon.i == i]
    bin_indices = (neuron_spikes/defaultclock.dt).astype(int)
    bin_indices = bin_indices[bin_indices < time_steps]
    binned_spikes[i, bin_indices] = 1
    
sampling_rate = 1 / defaultclock.dt
low_cut = freq_mod.base - 5 #come back to this and also review this code for conciseness
high_cut = freq_mod.base + 5

nyquist = 0.5 * sampling_rate
low = low_cut / nyquist
high = high_cut / nyquist
order = 4
b, a = butter(order, [low, high], btype='band')

print(f"Average PLV within the HPC: {singleRegionPLV(binned_spikes):.3f}")

# ------ pfc to pfc
binned_spikes2 = np.zeros((100, time_steps))

for i in range(100):
    neuron_spikes2 = spikemon_pfc.t[spikemon_pfc.i == i]
    bin_indices2 = (neuron_spikes2/defaultclock.dt).astype(int)
    bin_indices2 = bin_indices2[bin_indices2 < time_steps]
    binned_spikes2[i, bin_indices2] = 1

print(f"Average PLV within the PFC: {singleRegionPLV(binned_spikes2):.3f}")

#raster plot
plt.figure(figsize=(12,4))
plt.plot(spikemon.t/ms, spikemon.i, '.k', label="HPC")
plt.plot(spikemon_pfc.t/ms, spikemon_pfc.i, '.r', label="PFC")
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Raster Plot: HPC (black) + PFC (red)')
plt.legend()
plt.show()

#spectrogram

f1, t1, Sxx1 = spectrogram(rate, fs=1/(defaultclock.dt/second))
f_PFC, t_PFC, Sxx_PFC = spectrogram(rate_PFC, fs=1/(defaultclock.dt/second))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.pcolormesh(t1*1000, f1, Sxx1, shading='gouraud')
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

#heatmap (figure out how to do this properly as well)

#power spectrum
f, Pxx = welch(rate, fs=1/(defaultclock.dt/second), nperseg=1024)
f_PFC, Pxx_PFC = welch(rate_PFC, fs=1/(defaultclock.dt/second), nperseg=1024)

plt.figure(figsize=(10, 4))
plt.semilogy(f, Pxx, label='HPC')
plt.semilogy(f_PFC, Pxx_PFC, label='PFC')
plt.title('Power Spectrum (group 1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0,100)
plt.grid(True)
plt.legend()
plt.show()

#granger causality
data = pd.DataFrame({
    'HPC': rate,
    'PFC': rate_PFC
})

print('Granger Causality: HPC -> PFC')
grangercausalitytests(data[['PFC', 'HPC']], maxlag=10, verbose=True)
print('\nGranger Causality: PFC -> HPC')
grangercausalitytests(data[['HPC', 'PFC']], maxlag=10, verbose=True)

#mutual information
# ----- hpc pfc
bins = 20
hpc_binned = np.digitize(rate, np.histogram(rate, bins=bins)[1])
pfc_binned = np.digitize(rate_PFC, np.histogram(rate_PFC, bins=bins)[1])

mi = mutual_info_score(hpc_binned, pfc_binned)
print(f'Mutual Information (HPC - PFC): {mi:.3f} bits')

# ----- hpc hpc

print("Mutual Information Matrix (HPC):")
print(singleRegionMI(binned_spikes))

# ----- pfc pfc

print("Mutual Information Matrix (PFC):")
print(singleRegionMI(binned_spikes2))

#cognitive tasks
#pseudocode for task switching
#make two different poissongroup variables that represent different stimulus to the hpc that will result in different responses in the pfc
#make hpc neurons receive one or the other and fire patterns and pfc neurons trained to respond differently based on where the hpc activity came from
#set a trial duration and a number of trials (like 250)
#make a loop that runs through the number of trials and presents either stimulus 1, stimulus 2, or no stimulus (depending on schedule list) and maybe appends the results to a list you can average?
#generate task switching schedule by defining a function with parameters num_trials, switch_prob, and gap_length that contains a loop that adds 0,1, or 2 num_trials times based on whether a random 
#number is less than the probability of switching to determine if gaps should be added (according to gap length) + the value of the current task and append it to the schedule list
#record data by recording whether previously learned weights remain stable between tasks (record similarity between weights for relevant synapses across task phases. i.e. store weights before and after switch and calculate percent change)
#number of trials required for the network to adjust to a new task after a switch (if response after a switch is incorrect count the number of trials until accuracy reaches criterion)

#pseudocode for pattern matching
#define the number of patterns (like 5, 40 neurons per pattern). run a loop through each pattern to assign neurons to each pattern
#connect stimulation poissongroups with neurons with a connection parameter of j=i. create a network between the poissongroup and the synapses and run for 100ms before disconnecting.
#show each pattern many times to strengthen synapses between the hpc neurons in the pattern and the hpc response
#make a partial cue that is a subset of the pattern, present it, and see if the corresponding pfc neurons fire
#record how much overlap there is between the number of true positives and the target number. also record the false positive rate


#AD GROUP STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#disrupt oscillations/connections: impair synaptic plasticity by replacing previous synapses with new impaired ones, increase leak current/decrease excitability, progressively remove neurons
#ad group data
#ad group graphs/data collection
#ad group cognitive tasks
#ad group avg hamming distance with control group (convert neurons to array by binning spikes. two arrays will be compared)
#calculate correlations between hamming distance and percent change of synaptic weights in tasks + number of trials needed to adjust to new task

#OPEN LOOP GROUP STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#mutual info
