import numpy as np
from .utilities import ReLU

class MaxPooling(object):
	"""docstring for MaxPooling"""
	def __init__(self, filter_size:tuple, stride:int, ReLU:bool=True):
		# filter_size = (2,2), stride = 2
		super(MaxPooling, self).__init__()
		# self.batch_size = batch_size
		self.filter_size = filter_size
		self.stride = stride
		self.ReLU = ReLU

	def calc_output_size(self, input_size):
		n = len(input_size)
		output_size_ = int((input_size[-2] - self.filter_size[-2])/self.stride + 1)
		output_size = [input_size[i] for i in range(n-2)] +[output_size_, output_size_]
		return tuple(output_size)

	def forward(self, x:np.array, mode='train'):
		# x: 4-d array
		self.cache = x
		output_size = self.calc_output_size(x.shape)
		y = np.zeros(output_size)
		for i in range(output_size[2]):
			for j in range(output_size[3]):
				r = i*self.stride
				c = j*self.stride
				x_local = x[:,:, r:r+self.filter_size[0], c:c+self.filter_size[0]]
				y[:,:,i,j] = np.max(x_local ,axis=(2,3))
		if self.ReLU:
			y = ReLU(y)
		if mode == 'test':
			self.cache = []
		return y					
		
	def back(self, grad):
		# return dL/dX
		# grad = dL/dY
		x = self.cache
		output_size = self.calc_output_size(x.shape)
		J = np.zeros_like(x).astype(float)
		y = np.zeros(output_size)
		for i in range(output_size[2]):
			for j in range(output_size[3]):
				r = i*self.stride
				c = j*self.stride
				x_local = x[:,:, r:r+self.filter_size[0], c:c+self.filter_size[0]]
				x_local = np.transpose(x_local,(2,3,0,1))
				f = (x_local == np.max(x_local, axis=(0,1))).astype(float)
				f = np.transpose(f,(2,3,0,1))
				J[:,:,r:r+self.filter_size[0],c:c+self.filter_size[0]] += grad[:,:,i,j][:,:,np.newaxis,np.newaxis] * f

		if self.ReLU:
			J = J * (x>0).astype(float)
		return J

if __name__ == '__main__':
	X = np.array([
	    [6,2,5,4,4],
	    [9,1,5,3,7],
	    [7,2,4,3,8],
	    [4,5,5,1,0],
	    [0,2,8,0,8]
	])
	X = np.array([
		[X,-X],
		[2*X,X-7]
		])

	maxp = MaxPooling(filter_size=(3,3),stride=1,ReLU = True)
	Y = maxp.forward(X)
	print(Y)

	dLdY = np.array([
	    [0.1111,-0.0007,0],
	    [0,0.1104,0.1111],
	    [0.1111,0.1111,0.1111]
	])
	dLdY = np.array([
		[dLdY,dLdY],
		[dLdY,dLdY]
		])

	J = maxp.back(dLdY)

	print(J)
		