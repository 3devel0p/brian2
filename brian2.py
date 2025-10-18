%matplotlib inline
!pip install brian2 numpy scipy matplotlib statsmodels scikit-learn pyspike elephant neo quantities
#imports, variables, and function definitions
from brian2 import *
import numpy as np
import pyspike as spk
import neo
import quantities as pq
import matplotlib.pyplot as plt
import elephant.spike_train_dissimilarity as std
from scipy.signal import spectrogram, welch
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import hamming
from scipy.signal import hilbert, filtfilt, butter
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
from sklearn.metrics import mutual_info_score
duration = 1*second
#temporal time window for stdp ltp to occur is 20ms as observed by Silva et al. 2010 (https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2010.00012/full)
taupre = taupost = 20*ms
#normal window for stdp ltp to occur in pfc is 10ms (so it has slower learning) as observed by Ruan et al. 2014 (https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2014.00038/full)
taupre_pfc = taupost_pfc = 10*ms
Apre = 0.01
epsilon = 1e-10
Apost = -Apre*0.95
wmax = 1.0
tmax = 50*ms
freq_mod = 10 * Hz
time_steps = int(duration / defaultclock.dt)
numneurons = 100

def singleRegionPLV(spikebin):
    all_phases = np.zeros((numneurons, time_steps))

    for i in range(numneurons):
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
    mi_matrix = np.zeros((numneurons, numneurons))

    for i in range(numneurons):
        for j in range(i, numneurons): 
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

def create_bin(spikemon):
    binned_spikes = np.zeros((numneurons, time_steps)) 

    for i in range(numneurons):
        neuron_spikes = spikemon.t[spikemon.i == i]
        bin_indices = (neuron_spikes/defaultclock.dt).astype(int)
        bin_indices = bin_indices[bin_indices < time_steps]
        binned_spikes[i, bin_indices] = 1

    return binned_spikes

#function contains all graphs
def data_collection(pfc_population, hpc_population, synapse, synapse2, synapse3, synapse4, hpc_statemon, pfc_statemon, hpc_spikemon, pfc_spikemon):
    global b
    global a
    fs = 1 / (defaultclock.dt/second) / 10
    rate = hpc_population.smooth_rate(window='flat', width=5*ms)/Hz
    rate_PFC = pfc_population.smooth_rate(window='flat', width=5*ms)/Hz

    fft1 = fft(rate)
    fft_pfc = fft(rate_PFC)
    freqs = fftfreq(len(rate), 1/fs)

    #synaptic connections scatterplot that includes all four synapse groups
    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    plt.scatter(synapse.i, synapse.j, s=synapse.w*10, c=synapse.w, cmap='viridis')
    plt.xlabel('Pre-synaptic Neuron Index')
    plt.ylabel('Post-synaptic Neuron Index')
    plt.title('Synaptic Connections (HPC -> HPC)')

    plt.subplot(2,2,2)
    plt.scatter(synapse2.i, synapse2.j, s=synapse2.w*10, c=synapse2.w, cmap='viridis')
    plt.xlabel('Pre-synaptic Neuron Index')
    plt.ylabel('Post-synaptic Neuron Index')
    plt.title('Synaptic Connections (HPC -> PFC)')

    plt.subplot(2,2,3)
    plt.scatter(synapse3.i, synapse3.j, s=synapse3.w*10, c=synapse3.w, cmap='viridis')
    plt.xlabel('Pre-synaptic Neuron Index')
    plt.ylabel('Post-synaptic Neuron Index')
    plt.title('Synaptic Connections (PFC -> PFC)')

    plt.subplot(2,2,4)
    plt.scatter(synapse4.i, synapse4.j, s=synapse4.w*10, c=synapse4.w, cmap='viridis')
    plt.xlabel('Pre-synaptic Neuron Index')
    plt.ylabel('Post-synaptic Neuron Index')
    plt.title('Synaptic Connections (PFC -> HPC)')

    plt.colorbar(label='Synaptic Weight')
    plt.tight_layout()
    plt.show()

    #voltage traces (5% of hpc)
    plt.figure(figsize=(8,4))
    for i in range (int(numneurons/20)):
        plt.plot(hpc_statemon.t/ms, hpc_statemon.v[i], label=f'Neuron {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (v)')
    plt.title('Voltage Traces (HPC)')
    plt.legend()
    plt.show()

    #voltage traces (5% of pfc)
    plt.figure(figsize=(8,4))
    for i in range (int(numneurons/20)):
        plt.plot(pfc_statemon.t/ms, pfc_statemon.v[i], label=f'Neuron {i}')
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
    plt.xlim(0,100)
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

    binned_spikes = np.zeros((numneurons, time_steps))

    for i in range(numneurons):
        neuron_spikes = hpc_spikemon.t[hpc_spikemon.i == i]
        bin_indices = (neuron_spikes/defaultclock.dt).astype(int)
        bin_indices = bin_indices[bin_indices < time_steps]
        binned_spikes[i, bin_indices] = 1
        
    sampling_rate = 1 / defaultclock.dt
    low_cut = freq_mod.base - 5
    high_cut = freq_mod.base + 5

    nyquist = 0.5 * sampling_rate
    low = low_cut / nyquist
    high = high_cut / nyquist
    order = 4
    b, a = butter(order, [low, high], btype='band')

    print(f"Average PLV within the HPC: {singleRegionPLV(binned_spikes):.3f}")

    # ------ pfc to pfc
    binned_spikes2 = np.zeros((numneurons, time_steps))

    for i in range(numneurons):
        neuron_spikes2 = pfc_spikemon.t[pfc_spikemon.i == i]
        bin_indices2 = (neuron_spikes2/defaultclock.dt).astype(int)
        bin_indices2 = bin_indices2[bin_indices2 < time_steps]
        binned_spikes2[i, bin_indices2] = 1

    print(f"Average PLV within the PFC: {singleRegionPLV(binned_spikes2):.3f}")

    #raster plot
    plt.figure(figsize=(12,4))
    plt.plot(hpc_spikemon.t/ms, hpc_spikemon.i, '.k', label="HPC")
    plt.plot(pfc_spikemon.t/ms, pfc_spikemon.i, '.r', label="PFC")
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

def task_switch(trial, schedule, task1, task2): 
    #trial = int(t/defaultclock.dt / (100*ms/defaultclock.dt)) 
    #hpc receives a stimulus based on what trial
    current_task = schedule[trial]
    if current_task == 1:
        task1.rates = 10*Hz
        task2.rates = 0*Hz
    elif current_task == 2:
        task1.rates = 0*Hz
        task2.rates = 10*Hz
    else:
        task1.rates = 0*Hz
        task2.rates = 0*Hz

def teach(trial, schedule, pfcspikemon, pfc):
    #pfc learns a response accordingly. so, when the hpc gets the stimulus, the pfc learns to respond based on that.
    #trial = int(t/(100*ms))
    global trials_needed_before_correct
    global efficiency
    current_task = schedule[trial]
    if current_task == 1:
        pfc.v[:50] += 0.5
    elif current_task == 2:
        pfc.v[50:] += 0.5
    #record the number of trials needed for the right neurons to give a stronger response
    if current_task == 1 and np.sum(pfcspikemon.count[:50]) > np.sum(pfcspikemon.count[50:]): #figure out a better way to assess whether the right neurons are firing
        efficiency.append(trials_needed_before_correct)
        trials_needed_before_correct = 0
    elif current_task == 2 and np.sum(pfcspikemon.count[50:]) > np.sum(pfcspikemon.count[:50]):
        efficiency.append(trials_needed_before_correct)
        trials_needed_before_correct = 0
    else:
        trials_needed_before_correct += 1

def generate_schedule(num_trials, switch_prob, gap_length):
    schedule = []
    current_task = 1
    for trial in range(num_trials):
        if np.random.rand() < switch_prob:
            schedule.extend([0]*gap_length)
            if len(schedule) >= num_trials:
                break
            current_task = 3 - current_task
        schedule.append(current_task)
    return schedule

def cognitive_task_test(spikeMonitor, schedule, region_pfc, region_hpc, S2):
    global efficiency
    #task swtiching = teach the pfc to associate a certain stimulus from the hpc with a certain response
    stimulus1 = PoissonGroup(numneurons, rates=10*Hz)
    stimulus2 = PoissonGroup(numneurons, rates=20*Hz)

    #connect the first stimulus to the hpc
    synapse_stimulus1 = Synapses(stimulus1, region_hpc, 'w : 1', on_pre='v_post += 0.1')
    synapse_stimulus1.connect(p=0.1)
    synapse_stimulus1.w = 0.5

    #connect the second stimulus to the hpc
    synapse_stimulus2 = Synapses(stimulus2, region_hpc, 'w : 1', on_pre='v_post += 0.1')
    synapse_stimulus2.connect(p=0.1)
    synapse_stimulus2.w = 0.5

    weights_before_trials = []
    weights_after_trials = []
    efficiency = []
    list_of_percentages = []

    for trial in range(num_trials):
        weights_before_trials.append(np.mean(S2.w[:]).copy()) 
        task_switch(trial, schedule, stimulus1, stimulus2) 
        teach(trial,schedule,spikeMonitor, region_pfc)
        run(trial_duration)
        weights_after_trials.append(np.mean(S2.w[:]).copy())

    #calculate percent change in weights of synapses between hpc and pfc
    for i in range(len(weights_before_trials)):
        if any(weights_before_trials[i] == 0):
            print(f"Trial {i+1}: Percent Change = N/A (division by zero)")
            continue
        ratio = ((weights_after_trials[i] - weights_before_trials[i])/weights_before_trials[i]) * 100
        percent_change = round(ratio, 3)
        print(f"Trial {i+1}: Percent Change = {percent_change}%")
        list_of_percentages.append(percent_change)
    avg_percent_change = np.mean(list_of_percentages)
    print("Efficiency for the experimental condition: ", efficiency)
    print("Average: ", np.mean(efficiency))
    print("Standard deviation: ", np.std(efficiency))
    print("Average percent change in weights: ", avg_percent_change)
    print("Standard deviation of percent change in weights: ", np.std(list_of_percentages))
    return avg_percent_change
def find_vp_distance(exp_group_spikemon):
    control_spikes = [spikemon.t[spikemon.i==i]/ms for i in range(numneurons)] 
    exp_spikes = [exp_group_spikemon.t[exp_group_spikemon.i==i]/ms for i in range(numneurons)]
    cost = 1000.0 * pq.Hz #double check this
    vp_distances = []

    for i in range(numneurons):
        if len(control_spikes[i]) == 0 and len(exp_spikes[i]) == 0:
            continue
        st_control = neo.SpikeTrain(control_spikes[i] * pq.ms, t_stop=duration*1000) #repeat for all neurons
        st_exp = neo.SpikeTrain(exp_spikes[i] * pq.ms, t_stop=duration*1000)

        distance = std.victor_purpura_distance([st_control, st_exp], cost_factor=cost)
        vp_distances.append(distance)

    results = "Average Victor-Purpura Distance: " + str(np.mean(vp_distances))
    return results
def create_neurons_and_synapses():
    #lower threshold and tau is because hpc is more excitable and more glutamatergic than cerebral cortex according to Heckers & Konradi 2014 (https://pmc.ncbi.nlm.nih.gov/articles/PMC4402105/)
    hpc = NeuronGroup(numneurons, eqs, threshold='v>hpc_thresh', reset='v = 0', refractory=15*ms, method='euler')
    pfc = NeuronGroup(numneurons, eqs, threshold='v>pfc_thresh', reset='v = 0', refractory=15*ms, method='euler')
    hpc.v = 'rand()'
    hpc.tau = 10*ms
    pfc.v = 'rand()'
    pfc.tau = 20*ms

    #larger weight = stronger synaptic connections in hpc because of higher learning rate according to Guerreiro & Clopath, 2024 (https://www.biorxiv.org/content/10.1101/2024.02.01.578356v1.full)
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
    S2.w = 0.05 + 0.01*rand(len(S2))
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

    #probability and weight are lower here because pfc to hpc connections are sparse according to Malik et al. 2022 (https://www.sciencedirect.com/science/article/pii/S009286742200397X)
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
    return hpc, pfc, S, S2, S3, S4
#CONTROL DATA NEURON STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

v0_max = 3
v0_max_pfc = 1.5
sigma = 0.2
eqs = '''
dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
v0 : 1
tau : second
I_ext : volt
'''
hpc_thresh = 0.7
pfc_thresh = 0.8

hpc_control, pfc_control, S_control, S2_control, S3_control, S4_control = create_neurons_and_synapses()

M = StateMonitor(hpc_control, 'v', record=True)
M_PFC = StateMonitor(pfc_control, 'v', record=True)
spikemon = SpikeMonitor(hpc_control)
spikemon_pfc = SpikeMonitor(pfc_control)
pop = PopulationRateMonitor(hpc_control)
pop_pfc = PopulationRateMonitor(pfc_control)

hpc_control.v0 = 'i*v0_max/(N-1)'
pfc_control.v0 = 'i*v0_max_pfc/(N-1)'

#run(duration)
#data_collection(pop_pfc, pop, S, S2, S3, S4, M, M_PFC, spikemon, spikemon_pfc)


#cognitive tasks

trial_duration = 100*ms
num_trials = 10 #chanfe this to 100
trials_needed_before_correct = 0

#four experimental conditions
schedule_frequent_short = generate_schedule(num_trials, 0.6, 2)
schedule_frequent_long = generate_schedule(num_trials, 0.6, 7)
schedule_infrequent_short = generate_schedule(num_trials, 0.3, 2)
schedule_infrequent_long = generate_schedule(num_trials, 0.3, 7)
'''

#first experimental condition
#cognitive_task_test(spikemon_pfc, schedule_frequent_short, pfc_control, hpc_control, S2_control) #make sure to comment out each condition when running the other ones

#second experimental condition
#cognitive_task_test(spikemon_pfc, schedule_frequent_long, pfc_control, hpc_control, S2_control)

#third experimental condition
#cognitive_task_test(spikemon_pfc, schedule_infrequent_short, pfc_control, hpc_control, S2_control)
#fourth experimental condition
#cognitive_task_test(spikemon_pfc, schedule_infrequent_long, pfc_control, hpc_control, S2_control)



#AD GROUP STARTS HERE (remember synapses can be modified but monitors must be new)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
start_scope()
hpc_thresh = 0.8 #increases threshold/decreases excitability
pfc_thresh = 0.9
hpc_ad, pfc_ad, S_ad, S2_ad, S3_ad, S4_ad = create_neurons_and_synapses()
#citations can be found later for the comments below
magnitude = 0.7
#decreases weight of synapses (add citation)
S_ad.w *= magnitude
S2_ad.w *= magnitude
S3_ad.w *= magnitude
S4_ad.w *= magnitude
eqs += "\nI_noise = sigma * xi : volt" #adds noise
S_ad.pre.code = '''
v_post += w
w = clip(w + 0.5*Apre, 0, wmax)
'''
S2_ad.pre.code = '''
v_post += w
w = clip(w + 0.5*Apre, 0, wmax)
'''
S3_ad.pre.code = '''
v_post += w
w = clip(w + 0.5*Apre, 0, wmax)
'''
S4_ad.pre.code = '''
v_post += w
w = clip(w + 0.5*Apre, 0, wmax)
'''
hpc_ad.tau = 5*ms #decrease membrane time constant
pfc_ad.tau = 10*ms
#numneurons = 80 (removing neurons isn't working for now so we'll fix it later)
S_ad.connect(condition="i!=j", p=0.15) #decrease probability of connections
S2_ad.connect(condition="i!=j", p=0.05)
S3_ad.connect(condition="i!=j", p=0.15)
S4_ad.connect(condition="i!=j", p=0.015)
#stim = TimedArray([0,1,0,1] * mV, dt=1*ms)
#hpc.I_ext = stim(defaultclock.t) 
#pfc.I_ext = stim(defaultclock.t)

#new monitors for ad group
print("AD GROUP DATA")
M_ad = StateMonitor(hpc_ad, 'v', record=True)
M_PFC_ad = StateMonitor(pfc_ad, 'v', record=True)
spikemon_ad = SpikeMonitor(hpc_ad)
spikemon_pfc_ad = SpikeMonitor(pfc_ad)
pop_ad = PopulationRateMonitor(hpc_ad)
pop_pfc_ad = PopulationRateMonitor(pfc_ad)

#ad group graphs/data
#run(duration)
#data_collection(pop_pfc_ad, pop_ad, S, S2, S3, S4, M_ad, M_PFC_ad, spikemon_ad, spikemon_pfc_ad)

#ad group cognitive tasks
trials_needed_before_correct = 0
#cognitive_task_test(spikemon_pfc_ad, schedule_frequent_short, pfc_ad, hpc_ad, S2_ad)
#cognitive_task_test(spikemon_pfc_ad, schedule_frequent_long, pfc_ad, hpc_ad, S2_ad)
#cognitive_task_test(spikemon_pfc_ad, schedule_infrequent_short, pfc_ad, hpc_ad, S2_ad)
#cognitive_task_test(spikemon_pfc_ad, schedule_infrequent_long, pfc_ad, hpc_ad, S2_ad)

#victor-purpura distance
#print(find_vp_distance(spikemon_ad))
'''


#OPEN LOOP GROUP STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#do NOT create new neurons and synapses because this represents the ad group with stimulation
#inject currents to simulate open loop oscillations. come back to explain how this alleviates cognitive decline
theta = 6*Hz
gamma = 40*Hz
alpha = 10*Hz
beta = 20*Hz

I_theta = 1.5*mV * sin(2*pi*theta*defaultclock.t)
#defaultclock.t is time elapsed
I_gamma = 1*mV * sin(2*pi*gamma*defaultclock.t)
I_alpha = 0.5*mV * sin(2*pi*alpha*defaultclock.t)
I_beta = 0.5*mV * sin(2*pi*beta*defaultclock.t)

I_ext = I_beta + I_alpha + I_theta + I_gamma 
theta_phase = 2*pi*theta*defaultclock.t

gamma_modulation_strength = 0.5
I_ext = I_theta + (1 + gamma_modulation_strength*sin(theta_phase)) * I_gamma
#alpha and beta are pfc, theta and gamma are hpc
hpc.run_regularly('I_ext = I_theta + (1 +0.5*sin(2*pi*theta*t)) * I_gamma', dt=defaultclock.dt)
pfc.run_regularly('I_ext = I_alpha + (1 +0.5*sin(2*pi*alpha*t)) * I_beta', dt=defaultclock.dt)

#add monitors
print("OPEN LOOP GROUP DATA")
M_open = StateMonitor(hpc, 'v', record=True)
M_PFC_open = StateMonitor(pfc, 'v', record=True)
spikemon_open = SpikeMonitor(hpc)
spikemon_pfc_open = SpikeMonitor(pfc)
pop_open = PopulationRateMonitor(hpc)
pop_pfc_open = PopulationRateMonitor(pfc)

#data collection

#run(duration)
#data_collection(pop_pfc_open, pop_open, S, S2, S3, S4, M_open, M_PFC_open, spikemon_open, spikemon_pfc_open)'''

#CLOSED LOOP GROUP STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#design a neuromorphic algorithm that adjusts stimulation based on parameters of the brain and biomarkers
