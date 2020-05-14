import numpy as np
from utilities import *

class FullyConnected(object):
	"""docstring for FullyConnected"""
	def __init__(self, input_size:tuple,batch_size:int, channels:int, activation='ReLU'):
		super(FullyConnected, self).__init__()
		self.input_size = input_size
		self.batch_size = batch_size
		self.channels = channels
		self.activation = activation
		self.random =np.random.RandomState(None)
		self.t = 0
		self.init_params()
		self.cache = []

	def __str__(self):
		return str({
		"input_size": self.input_size,
		"batch_size": self.batch_size,
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
		W = np.zeros((self.channels,w_size))
		b = np.zeros(self.channels)
		for i in range(self.channels):
			W[i,:] = self.random.normal(loc=0.0, scale=0.1, size=w_size)
		self.params = [W,b]

	def forward(self,X):
		self.cache = [X]
		# y = x.reshape(-1)
		W = self.params[0]
		b = self.params[1].reshape(-1,1)
		Y = W@X + b
		grad_dX = W
		if self.activation=='ReLU':
			Y = ReLU(Y)
			grad_dX = (Y>0).T @ grad_dX
		elif self.activation=='sigmoid':
			Y = sigmoid(Y)
			grad_dX = (Y * (1-Y)).T  @ grad_dX
		self.cache.append(Y)

		return 	Y, grad_dX

	def Jacobian(self):
		X = self.cache[0]
		Y = self.cache[1]
		if self.activation=='ReLU':
			Jac = (Y > 0) @ X.T 
		if self.activation=='sigmoid':
			Jac = (Y * (1-Y)) @ X.T
		return Jac

	def update(self, grad:np.array, eta:float):
		J = self.Jacobian()
		delta_W = eta * sum(grad.reshape(-1)) * J # TODO
		delta_b = eta * sum(grad.reshape(-1))
		W,b = self.params
		W -= delta_W
		b -= delta_b
		self.params = [W,b]
		self.t += 1








		