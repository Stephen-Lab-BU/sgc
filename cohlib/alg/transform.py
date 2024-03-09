import numpy as np
# def get_real_idft(slen, J, fs, res, frange):
def construct_real_idft(slen, J, fs):
    n_elements = slen

    # k_low = int(np.ceil(frange[0]/fs*n_elements))
    # k_end = int(frange[1]/fs*n_elements)
    Jv = int(J*2)+2

    Wv = np.zeros((slen, Jv))
    for t in range(slen):
        for j in range(J+1):
            Wv[t,2*j] = np.cos(t*np.pi*j/J)
            Wv[t,2*j+1] = -np.sin((t*np.pi*j)/J)
    Wv = Wv * (2*np.pi/J)
    # if k_low == 0:
    Wv = np.delete(Wv, 1,1)
    return Wv

def generate_harmonic_dict(slen, fs, res, frange, widx=0):
	"""

	Generate dictionary with harmonic basis (not necessarily orthogonal)

	Inputs
	======
	slen: signal length
	fs: Sampling frequency
	res: desired spectral resolution
	frange: array_like. Frequency range
	widx: Window index. Defaults to 0

	Outputs
	=======
	d: dictionary
	oflag: indicator whether the dictionary is orthogonal or not
	"""

	assert(frange[0] >=0 and frange[1]>=0), "Non-negative frequency required"
	assert(frange[1] <= fs), "Has to be lower than the sampling frequency"

	n_elements = int(fs/res)

	k_low = int(np.ceil(frange[0]/fs*n_elements))
	k_end = int(frange[1]/fs*n_elements)

	freq = np.arange(k_low, k_end+1)/n_elements
	t = np.arange(widx*slen, (widx+1)*slen)

	omega = 2*np.pi*np.outer(t,freq)

	sincol = np.sin(omega)
	coscol = np.cos(omega)

	n_freqs = k_end-k_low+1

	d = np.zeros((slen, 2*n_freqs))
	for idx in np.arange(n_freqs):
		d[:, 2*idx] = coscol[:, idx]
		d[:, 2*idx+1] = sincol[:, idx]

	# Remove the zero column
	if k_low ==0:
		d = np.delete(d,1,1)

	norm = np.linalg.norm(d, 2, 0)

	d = np.divide(d, norm)

	# inner = np.matmul(d.T, d)
	# matsum = np.sum(abs(inner - np.diag(np.diag(inner))))
	# if matsum > 1e-7:
	# 	oflag = 0
	# else:
	# 	oflag = 1

	# # TODO still need to work this out analytically...
	# return d[:, 1:]
	return d

