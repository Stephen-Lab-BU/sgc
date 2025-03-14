import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib 


def plot_eigvals_em_iters(ax, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None):
    eigs = jnp.array([jnp.linalg.eigh(gamma_init[j_ind_full,:,:])[0]] + [jnp.linalg.eigh(gamma_r[j_ind,:,:])[0]  for gamma_r in gamma_iter_list])
    color = plt.cm.rainbow(jnp.linspace(0, 1, len(eigs)))
    for i, e in enumerate(eigs):
        ax.plot(e[::-1], color=color[i], linewidth=1)

def get_eigval(mat, rank):
    eigvals = jnp.linalg.eigh(mat)[0]
    return eigvals[-rank]

def get_eigvec(mat, rank):
    eigvecs = jnp.linalg.eigh(mat)[1]
    return eigvecs[:,-rank]

def plot_cross_spec_eigval_em_iters(ax, eigrank, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None, 
        color='tab:blue', style='-', width=2):
    eigs = jnp.array([get_eigval(gamma_init[j_ind_full,:,:], eigrank)] + [get_eigval(gamma_r[j_ind,:,:], eigrank) for gamma_r in gamma_iter_list])
    ax.plot(eigs, color=color, linewidth=width, linestyle=style)

def get_eigvec_em_iters(eigrank, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None):
    eigvecs = jnp.array([get_eigvec(gamma_init[j_ind_full,:,:], eigrank)] + [get_eigvec(gamma_r[j_ind,:,:], eigrank) for gamma_r in gamma_iter_list])
    return eigvecs

def plot_eigvec_func_em_iters(ax, func, eigrank, dim, gamma_iter_list, gamma_init, j_ind=9, nz=None, color='tab:blue'):
    eigvecs = get_eigvec_em_iters(eigrank, gamma_iter_list, gamma_init, j_ind, nz=nz)
    res = func(eigvecs[:,dim])
    ax.plot(res, color=color, linewidth=2)

def plot_cross_spec_em_iters(ax, i, j, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None):
    cs_real = jnp.array([gamma_init[j_ind_full,i,j].real] + [gamma_r[j_ind,i,j].real for gamma_r in gamma_iter_list])
    cs_imag = jnp.array([gamma_init[j_ind_full,i,j].imag] + [gamma_r[j_ind,i,j].imag for gamma_r in gamma_iter_list])
    ax.plot(cs_real, color='tab:blue', linewidth=2)
    ax.plot(cs_imag, color='tab:red', linewidth=2)

def plot_cross_spec_func_em_iters(ax, func, i, j, gamma_iter_list, gamma_init, j_ind=9, j_ind_full=9, nz=None, color='tab:blue', style='-', width=2):
    cs_real = jnp.array([func(gamma_init[j_ind_full,i,j])] + [func(gamma_r[j_ind,i,j]) for gamma_r in gamma_iter_list])
    ax.plot(cs_real, color=color, linewidth=width, linestyle=style)


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

    # if override_bg_color:
    #     ax.axhspan(0, n_units + 1, facecolor=override_bg_color, alpha=alpha) #

    # else:
    #     ax.axhspan(0, n_units + 1, facecolor=colors[region], alpha=alpha) #

    spikeraster = ax.imshow(spike_mat, origin=origin, **kwargs)
    spikeraster.set_clim(0., jnp.max(spike_mat) * contrast_scale)


