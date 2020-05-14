import numpy as np
from .utilities import *

class Convolution(object):
	"""docstring for Convolution"""
	def __init__(self, input_size:tuple, batch_size:int, filter_size:tuple, filter_count:int, stride:int=1, pad:int=0, ReLU=False):
		# input size: 3-d, filter size: 3-d
		# 3 * 32 * 32, 3 * 5 * 5
		super(Convolution, self).__init__()
		self.input_size = input_size
		self.batch_size = batch_size
		self.filter_size = filter_size
		self.filter_count = filter_count
		self.stride = stride
		self.pad = pad
		self.padding = padding
		self.ReLU = ReLU
		self.random =np.random.RandomState(None)
		self.init_params()
		self.cache = []
		self.t = 0
	
	def __str__(self):
		return str({
			"input_size": self.input_size,
			"filter_size": self.filter_size,
			"filter_count": self.filter_count,
			"stride":self.stride,
			"pad": self.pad,
			"params":self.params,
			"t":self.t,
			"cache":self.cache
			})

	def check_input_size(self, X):
		# X:4-d array (batch, channel, x, y)
		return X.shape[1:] == self.input_size and X.shape[0] == self.batch_size
	# def check_filter_size(self):

	def calc_output_size(self):
		padded_l = self.input_size[1] + 2 * self.pad
		output_size_ = int((padded_l-self.filter_size[1])/ self.stride +1)
		return (self.batch_size, self.filter_count, output_size_, output_size_)

	def init_params(self):
		self.params = []
		W_size = [self.filter_count] + list(self.filter_size)
		W = self.random.normal(loc=0.0, scale=0.1, size=W_size)
		b = np.zeros(self.filter_count)
		self.params = [W,b]
			

	def forward(self, X):
		if not self.check_input_size(X):
			raise Exception("input size does not fit")
		padded = self.padding(X, self.pad)
		# output_size = self.calc_output_size()
		W = self.params[0]
		b = self.params[1]
		# for n_filter in range(self.filter_count):
		# 	w = self.params[0][n_filter]
		# 	b = self.params[1][n_filter]
		# 	y_hat[n_filter,:,:] = convolution(padded, w, self.stride) + b
		H = convolution(padded,W,self.stride) +b.reshape(1,-1,1,1)
		if self.ReLU:
			Y_hat = ReLU(H)
		self.cache = [padded, H, Y_hat]

		return Y_hat

# not used
	# def Jacobian_dw(self):
	# 	# returns dY/dW
	# 	# This is not needed to calculate dL/dW.
	# 	X = self.cache
	# 	filter_size = self.filter_size
	# 	stride = self.stride
		
	# 	N_r = int((X.shape[1] - filter_size[1])/stride + 1)
	# 	N_c = int((X.shape[2] - filter_size[2])/stride + 1)
	# 	filter_total_entry = filter_size[0] * filter_size[1] * filter_size[2]
	# 	J = np.zeros((N_r * N_r, filter_total_entry))
	# 	for i in range(N_r):
	# 		for j in range(N_c):
	# 			for k in range(filter_total_entry):
	# 				X_local = X[:,i*stride:i*stride+filter_size[1], j*stride:j*stride+filter_size[2]]
	# 				J[i*N_c+j, k] = X_local.reshape(-1)[k]
	# 	return J
	
	def get_Jacobian(self, grad:np.array):
		# grad = dL/dy, 4-d (batch, filter_count, dim1, dim2)
		# dW = X_padded conv grad
		X_padded, H = self.cache[:2]
		if self.ReLU:
			dYdH = (H>0).astype(float)
			grad = grad * dYdH
		W = self.params[0]
		reverse_W = reverse_kernel(W)
		grad_padded = padding(grad, self.pad)
		dW = np.zeros_like(W)
		dX = np.zeros(([self.batch_size]+list(self.input_size)))
		for f in range(W.shape[0]):
			for c in range(W.shape[1]):
				for b in range(self.batch_size):
					dW[f,c] += convolution_basic(X_padded[b,c], grad[b,f], self.stride)
					dX[b,c] += convolution_basic(grad_padded[b,f], reverse_W[f,c], self.stride)
		# for b in range(self.batch_size):
		# 	for f in range(self.filter_count):
		# 		for c in range(W.shape[1]):
		# 			dX[b,c] += convolution_basic(grad_padded[b,f], reverse_W[f,c], self.stride)
		db = np.sum(grad,axis=(0,2,3))
		return dW, dX, db

	def update(self,dW,db,eta:float):
		W,b = self.params
		W -= eta * dW
		b -= eta * db
		self.params = [W,b]
		self.t += 1

if __name__ == '__main__':
	conv = Convolution(input_size=(1,4,4),batch_size=2, filter_size = (1,3,3),filter_count=1,pad=1)
	w = np.array(range(9)).reshape(1,3,3) * 0.1 + 0.1
	conv.params[0][0] = w
	X = np.array([[
	    [0,1,2,3],
	    [4,5,6,7],
	    [8,9,10,11],
	    [12,13,14,15]
	]])
	X = np.array([X,X])
	X_ = padding(X,1)
	print("padded X\n", X_)
	Y = conv.forward(X)
	print("Y\n",Y)
	# J = conv.Jacobian_dw()
	# print("Jacobian dY/dW\n", J)

	grad = np.array([[
	    [0.0625,0,0,0],
	    [0,.0625,0,0],
	    [0,0,0,.0625],
	    [.0625,.0625,0,0]
	]])
	grad = np.array([grad,grad])
	dW, dX, db = conv.get_Jacobian(grad)
	print("Jacobian dL/dW\n", dW)
	print("Jacobian dL/dX\n", dX)
	print("Jacobian dL/db\n", db)





