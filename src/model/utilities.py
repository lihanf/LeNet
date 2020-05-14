import numpy as np
from scipy.special import expit
MIN_THRESHOLD = 1E-10

def one_hot(y:np.array, n_class:int):
	Y = np.zeros((y.shape[0],n_class))
	for i in range(y.shape[0]):
	    Y[i,y[i]] = 1 
	return Y


def padding(x:np.array, pad:int):	
	# x: 4-d array
	if pad > 0:
		o = np.zeros((x.shape[0],x.shape[1], x.shape[2] + pad * 2, x.shape[3]+ pad *2))
		o[:,:,pad:-pad,pad:-pad] = x
		return o
	else:
		return x

def convolution(X:np.array,W:np.array, stride:int=1):
	# X 4-d array (batch, channel, dim1, dim2)
	# W 4-d array (filter_count, channel, dim1, dim2)
	N_r = int((X.shape[2] - W.shape[2])/stride + 1)
	N_c = int((X.shape[3] - W.shape[3])/stride + 1)
	H = np.zeros((X.shape[0],W.shape[0], N_r, N_c))
	for i in range(N_r):
		for j in range(N_c):
			X_local = X[:,:,i*stride:i*stride+W.shape[2], j*stride:j*stride+W.shape[3]]
			H[:,:,i,j] = np.tensordot(X_local,W,axes=([1,2,3],[1,2,3]))
	return H

# Not used
def convolution_basic(x:np.array,w:np.array, stride:int=1):
	# basic matrix convolution
	# x, w both 2-d array
	N_r = int((x.shape[0] - w.shape[0])/stride + 1)
	N_c = int((x.shape[1] - w.shape[1])/stride + 1)
	Y = np.zeros((N_r, N_c))
	for i in range(N_r):
		for j in range(N_c):
			X_local = x[i*stride:i*stride+w.shape[0], j*stride:j*stride+w.shape[1]]
			Y[i,j] = w.reshape(-1) @ X_local.reshape(-1)
	return Y


def reverse_kernel(W:np.array):
	W_r = np.zeros_like(W)
	n = W.shape[2]
	m = W.shape[3]
	for k in range(W.shape[0]):
		for i in range(W.shape[1]):
			w_t = W[k,i].reshape(-1)
			W_r[k,i,:,:] = np.flip(w_t, axis=0).reshape(n,m)
	return W_r

def cross_entropy(y, y_hat):
	# y_hat = y_hat.reshape(-1)
	# y_hat = y_hat.reshape(-1)
	y_hat[y_hat<MIN_THRESHOLD] = MIN_THRESHOLD
	y_hat[y_hat>1-MIN_THRESHOLD] = 1- MIN_THRESHOLD

	return - np.sum(y * np.log(y_hat))

def cross_entropy_grad(y, y_hat):
	# return dJ/dy_hat
	y_hat[y_hat<MIN_THRESHOLD] = MIN_THRESHOLD
	y_hat[y_hat>1-MIN_THRESHOLD] = 1- MIN_THRESHOLD
	# y_hat = y_hat.reshape(-1)
	return -y/y_hat + (1-y)/(1-y_hat)

def ReLU(x:np.array):
	return np.maximum(x,0)

def sigmoid(x:np.array):
	return expit(x)


def array_to_tex(X:np.array):
	# can only represent X must be 2-d array
	if len(X.shape) == 3:
		X = X.reshape(X.shape[1], X.shape[2])
	s = "\\begin{bmatrix}"
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			s += str(X[i,j])
			if j < X.shape[1] - 1:
				s += "&"
			else:
				if i < X.shape[0] - 1:
					s += "\\\\"
	s += "\\end{bmatrix}"
	return s
if __name__ == '__main__':
	W = np.arange(16).reshape(2,2,2,2)
	print(reverse_kernel(W))
