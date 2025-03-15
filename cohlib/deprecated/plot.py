import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

def draw_raster_single(spike_mat, trange=[0,0.5], color_name='k', contrast=0.1, ax=None):
    """
    Plotting function for raster image
    """
    ax = ax or plt.gca()
    contrast_scale = contrast
    origin = 'lower'

    color_rgb = matplotlib.colors.to_rgba(color_name)
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    #     color_name, [(0., 0., 0., 0.),  (0., 1., 0., 0.5)])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        color_name, [(0., 0., 0., 0.),  color_rgb])

    # colors = get_colors()
    kwargs =  {}
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

    # if override_bg_color:
    #     ax.axhspan(0, n_units + 1, facecolor=override_bg_color, alpha=alpha) #

    # else:
    #     ax.axhspan(0, n_units + 1, facecolor=colors[region], alpha=alpha) #

    spikeraster = ax.imshow(spike_mat, origin=origin, **kwargs)
    spikeraster.set_clim(0., np.max(spike_mat) * contrast_scale)
