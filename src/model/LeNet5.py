from .Convolution import Convolution
from .MaxPooling import MaxPooling
from .FullyConnected import FullyConnected
from .utilities import *

class LeNet5(object):
	"""docstring for LeNet5"""
	def __init__(self):
		super(LeNet5, self).__init__()
		# self.eta = eta 
		self.t = 0 

		self.Conv1 = Convolution(input_size=(1,32,32),filter_size=(1,5,5),filter_count=6)
		out_s = self.Conv1.calc_output_size()
		# print(out_s)

		self.MaxPool2 = MaxPooling(filter_size=(2,2), stride=2, ReLU=True)
		out_s = self.MaxPool2.calc_output_size(out_s)
		# print(out_s)

		self.Conv3 = Convolution(input_size=tuple(out_s), filter_size=(out_s[0],5,5),filter_count=16)
		out_s = self.Conv3.calc_output_size()
		# print(out_s)

		self.MaxPool4 = MaxPooling(filter_size=(2,2), stride=2, ReLU=True)
		out_s = self.MaxPool4.calc_output_size(out_s)
		# print(out_s)

		self.Conv5 = Convolution(input_size=tuple(out_s), filter_size=(out_s[0],5,5),filter_count=120, ReLU=True)
		out_s = self.Conv5.calc_output_size()
		# print(out_s)

		self.FC6 = FullyConnected(input_size=tuple(out_s),channels=84, activation='ReLU')
		out_s = (84,)
		self.FC7 = FullyConnected(input_size=out_s, channels=10, activation='softmax')

	def forward(self, X, mode):
		H1 = self.Conv1.forward(X,mode)
		H2 = self.MaxPool2.forward(H1,mode)
		H3 = self.Conv3.forward(H2,mode)
		H4 = self.MaxPool4.forward(H3,mode)
		H5 = self.Conv5.forward(H4,mode)
		H5 = H5.reshape(H5.shape[0],H5.shape[1])
		H6 = self.FC6.forward(H5,mode)
		Y_hat = self.FC7.forward(H6,mode)
		return Y_hat

	def Forward_Propagation(self, X, Y, mode:str='train'):
		Y_hat = self.forward(X, mode)
		Y_onehot = one_hot(Y,10)
		if mode == 'train':
			loss = cross_entropy(Y_onehot,Y_hat)
			self.cache = [X,Y_onehot,Y_hat]
			return loss
		elif mode == 'test':
			Y_pred = np.argmax(Y_hat,axis=1)
			err = len(Y[Y!=Y_pred])
			self.cache = []
			return err, Y_pred

	def get_Jacobian(self):
		X,Y_onehot,Y_hat = self.cache
		# grad = dJ/dY
		grad = cross_entropy_grad(Y_onehot,Y_hat)
		grad = grad.reshape(Y_hat.shape[0],10)
		return grad

	def Back_Propagation(self, eta):
		grad = self.get_Jacobian()
		dW7, dX7, db7, dH7 = self.FC7.back(grad,eta)

		dW6, dX6, db6, dH6 = self.FC6.back(dX7,eta)

		dX6 = dX6.reshape(dX6.shape[0],dX6.shape[1],1,1)
		dW5, dX5, db5 = self.Conv5.back(dX6,eta)

		dX4 = self.MaxPool4.back(dX5)

		dW3, dX3, db3 = self.Conv3.back(dX4,eta)

		dX2 = self.MaxPool2.back(dX3)
		dW1, dX1, db1 = self.Conv1.back(dX2,eta)

		self.t+=1 
		
if __name__ == '__main__':
	model = LeNet5()