import numpy as np
from scipy.special import softmax
from .utilities import *

class FullyConnected(object):
	"""docstring for FullyConnected"""
	def __init__(self, input_size:tuple, channels:int, activation='ReLU'):
		super(FullyConnected, self).__init__()
		self.input_size = input_size

		self.channels = channels
		self.activation = activation
		self.random =np.random.RandomState(None)
		self.t = 0
		self.init_params()
		self.cache = []

	def __str__(self):
		return str({
		"input_size": self.input_size,

		"channels": self.channels,
		"activation": self.activation,
		"t": self.t,
		"params":self.params,
		"cache":self.cache
		})

	def init_params(self):
		w_size = 1
		for s in self.input_size:
			w_size *= s
		# self.params = []
		b = np.zeros(self.channels)
		W = self.random.normal(loc=0.0, scale=.1, size=(w_size,self.channels))
		self.params = [W,b]

	def set_params(self,W,b):
		self.params = [W,b]


	def forward(self,X, mode='train'):
		W = self.params[0]
		b = self.params[1].reshape(1,-1)
		# print(W.shape, X.shape)
		H = X@W+ b
		if self.activation=='ReLU':
			Y = ReLU(H)
		elif self.activation=='sigmoid':
			Y = sigmoid(H)
		elif self.activation == 'softmax':
			Y = softmax(H, axis=1)
		self.cache = [X,H,Y]
		if mode == 'test':
			self.cache = []
		return 	Y

	# def Jacobian(self):
	# 	# returns dY/dw
	# 	X = self.cache[0]
	# 	Y = self.cache[1]
	# 	if self.activation=='ReLU':
	# 		Jac = (Y > 0) @ X.T 
	# 	if self.activation=='sigmoid':
	# 		Jac = (Y * (1-Y)) @ X.T
	# 	return Jac

	# In fact we do not need Jacobian dY/dW to calculate dW

	def back(self, grad:np.array,eta):
		X, H, Y = self.cache
		W,b = self.params
		if self.activation == 'ReLU':
			grad_dH = (H > 0).astype(float) * grad
		elif self.activation == 'sigmoid' or self.activation == 'softmax':
			grad_dH = (Y*(1.-Y)) * grad
		delta_W = X.T @ grad_dH
		delta_X = grad_dH @ W.T
		delta_b = np.sum(grad_dH,axis=0)
		W -= eta * delta_W
		b -= eta * delta_b
		self.params = [W,b]
		self.t += 1
		return delta_W, delta_X, delta_b, grad_dH

	# def update(self, delta_W, delta_b, eta:float):
	# 	W,b = self.params
	# 	W -= eta * delta_W
	# 	b -= eta * delta_b
	# 	self.params = [W,b]
	# 	self.t += 1









		