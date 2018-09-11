import math 
import numpy as np 
import h5py 
import random
import scipy
from scipy import ndimage
import tensorflow as tf 
from nltk import ConfusionMatrix
import matplotlib.pyplot as plt

class CNN_train(object):

	
	def __init__(self, conv_layers, flow, learning_rate, epochs, batch_size, 
		X_train, X_val, Y_train, Y_val, print_cost = True):

		"""
		The CNN_train class is initialized with its arguments when its 
		instance is created.
		"""

		tf.reset_default_graph()

		_, n_H0, n_W0, n_C0 = X_train.shape
		_, n_y = Y_train.shape
		self.X, self.Y = self._create_placeholder(n_H0, n_W0, n_C0, n_y)
		self.parameters = self._initialize_parameters(conv_layers)
		self.Z_n = self._forward_propagation(flow)
		self.cost = self._compute_cost()
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as sess:	

			sess.run(init)

			seed = 0
			epoch_plot = list()
			for epoch in range(epochs):

				epoch_cost = 0
				mini_batches = self._random_mini_batches(X_train, Y_train, batch_size, seed = seed)
				seed += 1

				for batch in mini_batches:

					X_mini, Y_mini = batch
					mini_batch_cost,_ = sess.run([self.cost, self.optimizer], feed_dict = 
						{self.X: X_mini, self.Y : Y_mini})

					epoch_cost += mini_batch_cost/len(mini_batches)

				if epoch % save == 0:
					save_path = saver.save(sess, './data/model.ckpt', global_step = epoch)
					print("Model saved in path: %s" % save_path)

				if print_cost == True and epoch % 10 == 0:
					print("Cost after %d epoch is : %6.2f" % (epoch, epoch_cost))
					print("Validation loss is : %6.2f"  % (sess.run(self.cost, feed_dict = 
						{self.X: X_val, self.Y : Y_val})))

				if epoch % 5 == 0:
					epoch_plot.append(epoch_cost)

			## Plotting the loss curve 
			plt.plot(epoch_plot)
			plt.ylabel('cost')
			plt.xlabel('iterations (per tens)')
			plt.title("Learning rate = " + str(learning_rate))
			plt.show()

			## Printing the train and validation accuracy
			correct_prediction = tf.equal(tf.argmax(self.Z_n,1), tf.argmax(self.Y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print("Train Accuracy : ", accuracy.eval({self.X:X_train, self.Y:Y_train}))
			print("Validation Accuracy : ", accuracy.eval({self.X:X_val, self.Y:Y_val}))


	def _create_placeholder(self, n_H0, n_W0, n_C0, n_y):

		"""
		Creates the placeholders for the tensorflow session

		Arguments:
		n_H0 -- scalar, height of the input image
		n_W0 -- scalar, width of the input image
		n_C0 -- scalar, number of channels of the input image
		n_y -- scalar, number of classes

		Returns:
		X -- placeholder for data input, of shape (None, n_H, n_W, n_C)
		Y -- placeholder for input labels (None, n_y)
		"""

		X = tf.placeholder(dtype = tf.float32, shape = (None, n_H0, n_W0, n_C0))
		Y = tf.placeholder(dtype = tf.float32, shape = (None, n_y))

		return X, Y

	
	def _initialize_parameters(self, conv_layers):

		"""
		Initializer the weight parameters to build the neural network

		Arguments:
		conv_layers -- vector, list of list with filter dimensions

		Returns:
		parameters -- dictionary of tensors containing initialized weights
		"""

		# tf.set_random_seed()  # important for reproducibility

		parameters = dict()

		for i in range(len(conv_layers)):

			parameters['W_{}'.format(i+1)] = tf.get_variable(name = 'W_{}'.format(i+1), 
				shape = conv_layers[i], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

		return parameters


	def _forward_propagation(self, flow):

		"""
		Implements the forward pass for the model. The steps would depend on
		the 'flow' input.

		Arguments:
		flow -- a dictionary indicating the strides and filters at every layer, and pooling 
		if required. 
		Eg: {'CONV2D_1':(s_c1,'SAME'),'CONV2D_2':(s_c2,'VALID'),'POOL_1':(s_p1,f_p1,'SAME'),
		'FC_1':(100,'relu'), 'FC_2':(6,'None')}. The dictionary indicates:
			- the stride and padding type at each convolution 
			- the stride and filter size and padding type at each pooling step
			- the number of neurons and the activation at each fully connected layer 
			- the dropout at after each convolution and pooling layer

		Returns:
		Z_n -- the output of the last linear unit 

		"""
		A_cnn = self.X # activations/input for convolutional layers

		for i in range(len(self.parameters.items())):

			W = self.parameters['W_{}'.format(i+1)]
			s, pad = flow['CONV2D_{}'.format(i+1)]
			Z = tf.nn.conv2d(input = A_cnn, filter = W, strides = [1,s,s,1], padding = pad)
			A_cnn = tf.nn.relu(Z)

			if 'POOL_{}'.format(i+1) in flow:
				s, f, pad, mode = flow['POOL_{}'.format(i+1)]
				if mode == 'max': A_cnn = tf.nn.max_pool(value = A_cnn, ksize = [1,f,f,1], 
					strides = [1,s,s,1], padding = pad)
				elif mode == 'avg': A_cnn = tf.nn.avg_pool(value = A_cnn, ksize = [1,f,f,1], 
					strides = [1,s,s,1], padding = pad)

			if 'DROP_{}'.format(i+1) in flow:
				keep_prob = flow['DROP_{}'.format(i+1)]
				A_cnn = tf.nn.dropout(A_cnn, keep_prob = keep_prob)


		i = 0
		A_fc = tf.contrib.layers.flatten(A_cnn) # activations/input for fully connected layers

		while 'FC_{}'.format(i+1) in flow:

			size, acti = flow['FC_{}'.format(i+1)]
			A_fc = tf.contrib.layers.fully_connected(inputs = A_fc, num_outputs = size, activation_fn = acti) 
			i += 1

		Z_n = A_fc
		return Z_n


	def _compute_cost(self):

		"""
		The cost is computed here

		Returns:
		Cost -- a tensor of teh cost function 
		"""

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.Z_n, labels = self.Y))

		return cost


	def _random_mini_batches(self, X, Y, batch_size, seed = 0):

		"""
		A helper function that aids with splitting the dataset into sets having
		a select batch size

		Arguments:
		X -- input data of shape (m, n_H, n_W, n_C)
		Y -- target data of shape (m, n_y)
		batch_size -- size of each mini-batch 
		seed -- introduce randomness into mini-batch construction

		Returns:
		mini_batches -- a list of mini-batches 
		"""

		m = X.shape[0]
		mini_batches = []
		np.random.seed(seed)

		random_index = list(np.random.permutation(m))
		X_shuffle = X[random_index, :, :, :]
		Y_shuffle = Y[random_index, :]

		for i in range(int(m/batch_size)):

			mini_batch_x = X_shuffle[i*batch_size : i*batch_size+batch_size, :, :, :]
			mini_batch_y = Y_shuffle[i*batch_size : i*batch_size+batch_size, :]
			mini_batch = (mini_batch_x, mini_batch_y)
			mini_batches.append(mini_batch)

		if m % batch_size != 0:

			mini_batch_x = X_shuffle[i*batch_size :, :, :, :]
			mini_batch_y = Y_shuffle[i*batch_size :, :]
			mini_batch = (mini_batch_x, mini_batch_y)
			mini_batches.append(mini_batch)

		return mini_batches




class CNN_infer(object):

	
	def __init__(self, X_test, conv_layers, flow, mode, path_saver, Y_test = None):

		"""
		The CNN_infer class is initialized with its arguments when its 
		instance is created.
		"""

		_, n_H0, n_W0, n_C0 = X_test.shape

		if mode == 'compare' : 
			_, n_y = Y_test.shape
			self.X, self.Y = self._create_placeholder(n_H0, n_W0, n_C0, n_y)

		elif mode == "predict": 
			self.X = tf.placeholder(dtype = tf.float32, shape = [None, n_H0, n_W0, n_C0])
		
		self.parameters = self._initialize_parameters(conv_layers)
		self.Z_n = self._forward_propagation(flow)
		
		saver = tf.train.Saver()

		with tf.Session() as sess:

			saver.restore(sess, path_saver)
			self.logits = sess.run(self.Z_n, feed_dict = {self.X:X_test})

			if mode == 'compare':

				correct_prediction = tf.equal(tf.argmax(self.Z_n, axis = 1), tf.argmax(self.Y, axis = 1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				print("Test Accuracy : ", accuracy.eval({self.X:X_test, self.Y:Y_test}))

				logits_argmax = np.argmax(self.logits, axis = 1)
				logits_argmax = [i for i in logits_argmax]
				y_argmax = np.argmax(Y_test, axis = 1)
				y_argmax = [i for i in y_argmax]
				print(ConfusionMatrix(y_argmax, logits_argmax))

			elif mode == 'predict':

				prediction = np.argmax(self.logits)
				print("Your algorithm predicts Y = " + str(prediction)) 


	def _create_placeholder(self, n_H0, n_W0, n_C0, n_y):

		"""
		Creates the placeholders for the tensorflow session

		Arguments:
		n_H0 -- scalar, height of the input image
		n_W0 -- scalar, width of the input image
		n_C0 -- scalar, number of channels of the input image
		n_y -- scalar, number of classes

		Returns:
		X -- placeholder for data input, of shape (None, n_H, n_W, n_C)
		Y -- placeholder for input labels (None, n_y)
		"""

		X = tf.placeholder(dtype = tf.float32, shape = (None, n_H0, n_W0, n_C0))
		Y = tf.placeholder(dtype = tf.float32, shape = (None, n_y))

		return X, Y


	def _initialize_parameters(self, conv_layers):

		"""
		Initializer the weight parameters to build the neural network

		Arguments:
		conv_layers -- vector, list of list with filter dimensions

		Returns:
		parameters -- dictionary of tensors containing initialized weights
		"""

		# tf.set_random_seed()  # important for reproducibility

		parameters = dict()

		for i in range(len(conv_layers)):

			parameters['W_{}'.format(i+1)] = tf.get_variable(name = 'W_{}'.format(i+1), 
				shape = conv_layers[i], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

		return parameters



	def _forward_propagation(self, flow):

		"""
		Implements the forward pass for the model. The steps would depend on
		the 'flow' input.

		Arguments:
		flow -- a dictionary indicating the strides and filters at every layer, and pooling 
		if required. 
		Eg: {'CONV2D_1':(s_c1,'SAME'),'CONV2D_2':(s_c2,'VALID'),'POOL_1':(s_p1,f_p1,'SAME'),
		'FC_1':(100,'relu'), 'FC_2':(6,'None')}. The dictionary indicates:
			- the stride and padding type at each convolution 
			- the stride and filter size and padding type at each pooling step
			- the number of neurons and the activation at each fully connected layer 
			- the dropout at after each convolution and pooling layer

		Returns:
		Z_n -- the output of the last linear unit 

		"""
		A_cnn = self.X # activations/input for convolutional layers

		for i in range(len(self.parameters.items())):

			W = self.parameters['W_{}'.format(i+1)]
			s, pad = flow['CONV2D_{}'.format(i+1)]
			Z = tf.nn.conv2d(input = A_cnn, filter = W, strides = [1,s,s,1], padding = pad)
			A_cnn = tf.nn.relu(Z)

			if 'POOL_{}'.format(i+1) in flow:
				s, f, pad, mode = flow['POOL_{}'.format(i+1)]
				if mode == 'max': A_cnn = tf.nn.max_pool(value = A_cnn, ksize = [1,f,f,1], 
					strides = [1,s,s,1], padding = pad)
				elif mode == 'avg': A_cnn = tf.nn.avg_pool(value = A_cnn, ksize = [1,f,f,1], 
					strides = [1,s,s,1], padding = pad)

		i = 0
		A_fc = tf.contrib.layers.flatten(A_cnn) # activations/input for fully connected layers

		while 'FC_{}'.format(i+1) in flow:

			size, acti = flow['FC_{}'.format(i+1)]
			A_fc = tf.contrib.layers.fully_connected(inputs = A_fc, num_outputs = size, activation_fn = acti) 
			i += 1

		Z_n = A_fc
		return Z_n




def load_preprocess(path, mode):

	"""
	This function loads and preprocess the data for sepecific needs.

	Arguments:
	path -- file path of the data tha ought to be processed
	mode -- the purpose the processed data is going to server : training, comparing or predicting

	Returns:
	X_train, X_val, Y_train, Y_val or X_test, Y_test or image_pred
	"""

	
	if mode == 'train':

		train_dataset = h5py.File(path, 'r')
		train_set_x_orig = np.array(train_dataset["train_set_x"][:])
		train_set_y_orig = np.array(train_dataset["train_set_y"][:])
		train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

		classes = np.array(train_dataset['list_classes'][:])
		val_idx = list()

		for classe in classes:

			temp_ls = [n for n,i in enumerate(list(train_set_y_orig[0,:])) if i == classe]
			val_idx.extend(random.sample(temp_ls,15))

		random.shuffle(val_idx)
		X_val = train_set_x_orig[val_idx]/255
		Y_val = np.eye(len(classes))[train_set_y_orig[0,val_idx].reshape(-1)]

		## train set
		idx = [i for i in range(train_set_y_orig.shape[1])]
		train_idx = [x for x in idx if x not in val_idx]
		X_train = train_set_x_orig[train_idx]/255
		Y_train = np.eye(len(classes))[train_set_y_orig[0,train_idx].reshape(-1)]

		return X_train, X_val, Y_train, Y_val


	elif mode == 'compare':

		test_dataset = h5py.File(path, 'r')
		test_set_x_orig = np.array(test_dataset["test_set_x"][:])
		test_set_y_orig = np.array(test_dataset["test_set_y"][:])
		test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

		classes = np.array(test_dataset['list_classes'][:])

		X_test = test_set_x_orig/255
		Y_test = np.eye(len(classes))[test_set_y_orig.reshape(-1)]

		return X_test, Y_test


	elif mode == 'predict':

		fname = path
		image = np.array(ndimage.imread(fname, flatten = False))
		image_pred = scipy.misc.imresize(image, size = (64,64)).reshape((1,64,64,3))

		#to plot : plt.imshow(image)

		return image_pred		

		## see an example
		# index = 1
		# plt.imshow(train_set_x_orig[index])
		# print("y = " + str(np.squeeze(train_set_y_orig[:,index])))



def main(mode, path, conv_layers, flow, epochs, batch_size, print_cost, learning_rate, path_saver):
	
	if mode == 'train':

		X_train, X_val, Y_train, Y_val = load_preprocess(path, mode)
		train_cnn = CNN_train(conv_layers, flow, learning_rate, epochs, batch_size, 
		X_train, X_val, Y_train, Y_val, print_cost)

	elif mode == 'compare':
		X_test, Y_test = load_preprocess(path, mode)
		infer_cnn = CNN_infer(X_test, conv_layers, flow, mode, path_saver, Y_test)
		

	elif mode == 'predict':
		X_pred = load_preprocess(path, mode)
		infer_cnn = CNN_infer(X_pred, conv_layers, flow, mode, path_saver)
		


if __name__ == '__main__':

	mode = 'compare' # train, compare and predict modes
	path = "msc/test_signs.h5"
	conv_layers = [[4, 4, 3, 8],[2, 2, 8, 16]]
	flow = {'CONV2D_1':(1,'SAME'), 'POOL_1':(8,8,'SAME','max'), 'CONV2D_2':(1,'SAME'), 
	'POOL_2':(4,4,'SAME','max'), 'FC_1':(6, None), 'DROP_1':0.9, 'DROP_2':1.0} # chk _forward_propagation() function for explanation
	epochs = 500 # can be ignored during comparing/predicting
	batch_size = 64 # can be ignored during comparing/predicting
	print_cost = True # can be ignored during comparing/predicting
	learning_rate = 0.001 # can be ignored during comparing/predicting
	save = 10 # saves model after every 10 epochs (can be ignored during comparing/predicting)
	path_saver = "msc/model.ckpt-470" # path of the saved model (can be ignored during trainig)

	main(mode, path, conv_layers, flow, epochs, batch_size, print_cost, learning_rate, path_saver)
	









