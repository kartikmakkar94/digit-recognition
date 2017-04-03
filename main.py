import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)	#one_hot means for 0 we have result as [1,0,0,0,0,0,0,0,0,0]

n_input_features = 784	#input is a 28x28 image, all the pixel values are unrolled in a vector

n_nodes_hl1 = 500	#no. of nodes in hidden layers of the Neural Network
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10	#no. of output classes

batch_size = 100	#no. of training examples in one forward and backward pass i.e Amount of data loaded in memory at a time

x = tf.placeholder('float', [None, n_input_features])	#placeholders for values, [None, 784] defines shape of tensor and throws an error if data is not of same dimension
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_input_features, n_nodes_hl1])),	#each layer is a dictionary having 2 tensors 
						'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
						'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)	#data is going through activation function

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

	return output

def train_neural_network(x):
	prediciton = neural_network_model(x)	#get predicitons from neural network
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediciton, labels= y))	#cost is synonymous with loss function

	optimizer = tf.train.AdamOptimizer().minimize(cost)	#default learning rate = 0.001, can be specified in AdamOptimizer

	hm_epochs = 10	#one epoch is forward and backward pass of all training samples

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())	#run the session

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', hm_epochs, ', loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediciton, 1), tf.argmax(y, 1))	#tells how many predictions were correct

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)



