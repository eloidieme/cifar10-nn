import numpy as np
import matplotlib.pyplot as plt

def montage(data, labels=None, nrows=5, ncols=5, savepath=None, show=True):
	""" Display the image for each label in W """
	label_list = [
		"airplane",
		"automobile",
		"bird",
		"cat",
		"deer",
		"dog",
		"frog",
		"horse",
		"ship",
		"truck"
    ]
	fig, ax = plt.subplots(nrows,ncols)
	for i in range(nrows):
		for j in range(ncols):
			im  = data[i*ncols+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			if labels: ax[i][j].set_title(label_list[labels[5*i+j]])
			ax[i][j].axis('off')
	plt.tight_layout()
	if savepath:
		plt.savefig(savepath)
	if show:
		plt.show()