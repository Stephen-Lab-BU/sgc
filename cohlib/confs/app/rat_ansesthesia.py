from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from pynwb import NWBHDF5IO

@dataclass
class AppRat15:
    data_name: str = 'app_rat15'
    data_path: str = '/projectnb/stephenlab/jtauber/cohlib/nwb/Rat15_Insertion1_Depth2.nwb'
    exp_type: str = 'topK'

def load_app_data(cfg):
    print('Loading App Data')
    acfg = cfg.app
    print(f'Experiment Type: {acfg.exp_type}')
    lcfg = cfg.latent

    io = NWBHDF5IO(acfg.data_path, mode="r")
    nwbfile = io.read()
    units = nwbfile.units

    lcfg.L
    lcfg.K
    win_duration = int(2*lcfg.num_freqs / 1000)

    duration = 10
    spikes_windows = []

    for i in range(lcfg.L):
        time_start = i*win_duration 
        time_end = time_start + duration
        trial_spikes = get_trial_spiketimes(units, time_start, time_end)
        spikes_window = mv_spike_times_to_binary(trial_spikes, time_start, duration)
        spikes_windows.append(spikes_window)

    spikes_all_units = jnp.stack(spikes_windows, axis=2)
    sort_fr = jnp.argsort(spikes_all_units.mean((0,2)))[::-1]
    sorted_spikes_all_units = spikes_all_units[:,sort_fr,:]

    if acfg.exp_type == 'topK':
        obs = sorted_spikes_all_units[:,:lcfg.K,:]
    elif acfg.exp_type == 'bottomK':
        obs = sorted_spikes_all_units[:,-lcfg.K:,:]
    else:
        raise NotImplementedError

    app_data = {'obs': obs}

    return app_data


    # sort units by firing rate
    # select units based on firing rates and lcfg.K
    # return dict with 'obs': selected_units 


def get_trial_spiketimes(units, time_start, time_end):

    num_units = len(units['spike_times'])
    trial_spikes = []
    for n in range(num_units):
        st_n = units['spike_times'][n]
        time_filt = (st_n > time_start) & (st_n < time_end)
        trial_spikes.append(st_n[time_filt])

    return trial_spikes

def mv_spike_times_to_binary(trial_spikestimes_mv, time_start, duration):
    K = len(trial_spikestimes_mv)
    units_binary = []
    for k in range(K):
        binary_k = spike_times_to_binary(trial_spikestimes_mv[k], time_start, duration=10)
        units_binary.append(binary_k)

    return np.stack(units_binary, axis=1)

def spike_times_to_binary(spike_times, time_start, duration=None, bin_width=0.001):
    """
    Convert spike times to a binary vector with specified bin width.

    Parameters
    ----------
    spike_times : array-like
        List or array of spike times (in seconds).
    duration : float, optional
        Total duration to cover (in seconds). 
        If None, it uses the maximum spike time rounded up.
    bin_width : float, optional
        Width of each bin (in seconds). Default is 0.001 (1 ms).

    Returns
    -------
    binary_vector : np.ndarray
        Binary vector indicating spike occurrence in each bin.
    """
    spike_times = np.asarray(spike_times) - time_start
    
    if duration is None:
        duration = np.ceil(spike_times.max() / bin_width) * bin_width
    
    n_bins = int(np.ceil(duration / bin_width))
    binary_vector = np.zeros(n_bins, dtype=int)
    
    spike_indices = (spike_times / bin_width).astype(int)
    spike_indices = spike_indices[spike_indices < n_bins]  # avoid out-of-bounds
    binary_vector[spike_indices] = 1
    
    return binary_vector
