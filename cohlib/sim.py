import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def draw_spikes(x, mu, beta):
    lamb = 1 / (1 + np.exp(-(mu + beta*x)))
    spikes = np.random.binomial(1, lamb)
    return spikes
                
def spikes_from_xns(x, pp_params, n_trials, T):
    mu = pp_params['mu']
    beta = pp_params['beta']
    C = mu.size
    spikes = np.zeros((n_trials, C, T))
    for l in range(n_trials):
        spikes_l = np.stack([draw_spikes(x[l,:], mu[c], beta[c]) for c in range(C)])
        spikes[l,:,:] = spikes_l
    return spikes



def draw_raster_single(spike_mat, trange, region, contrast=0.1, ax=None, ylabel=True,
                        override_bg_color=False, origin='lower', **kwargs):
    ax = ax or plt.gca()
    contrast_scale = contrast
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'gray_r_a', [(0., 0., 0., 0.),  (0., 0., 0., 1.)])
    
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('cmap', cmap)
    kwargs.setdefault('interpolation', None)
    kwargs.setdefault('extent', (trange[0], trange[-1], 0., spike_mat.shape[0]))
    
    ymin = 0
    ymax = spike_mat.shape[0]
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel('Spikes')
    ax.get_yaxis().set_ticks([])
    ax.tick_params(axis='x', which='major', labelsize=12)
    
    alpha = 0.2
    n_units = spike_mat.shape[0]

    if override_bg_color:
        ax.axhspan(0, n_units + 1, facecolor=override_bg_color, alpha=alpha) #

    else:
        ax.axhspan(0, n_units + 1, facecolor=colors[region], alpha=alpha) #
    
    spikeraster = ax.imshow(spike_mat, origin=origin, **kwargs)
    spikeraster.set_clim(0., np.max(spike_mat) * contrast_scale)
    
    
    return spikeraster
