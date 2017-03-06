# Get Dataset
from tensorflow.examples.tutorials.mnist import input_data
# note url get fails, direct download from:https://web.archive.org/web/20160117040036/http://yann.lecun.com/exdb/mnist/
#and place in: C:\Program Files\Anaconda3\Lib\site-packages\tensorflow\examples\tutorials\mnist
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

import tensorflow as tf

# set hyperparameters
learning_rate = 0.01 # How fast we update our weights
training_iteration = 30
batch_size = 100
display_step = 2

# graph input
x = tf.placeholder("float", [None, 784]) #mnist images of shape 28*28=784
y = tf.placeholder("float", [None,10]) # 0-9 digits

# set the weights
W = tf.Variable(tf.zeros([784, 10]))# weights are probability that affects how data flows in graph, updated continusly
b = tf.Variable(tf.zeros([10])) # fits the regression line to better fit the data

with tf.name_scope("Wx_b") as scope: # scopes help organize nodes in visualiser
    # linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)

# summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# more namescopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    # minimize error using cross entropy
    # cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# initializing the variables
init = tf.global_variables_initializer()

# merge all summaries into a single oeprator
merged_summary_op = tf.summary.merge_all()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # set the logs writer to the folder
    summary_writer = tf.summary.FileWriter("\logs", sess.graph)

    # training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # write logs for each interation
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch+i)
        # display logs per iteration step
        if iteration % display_step ==0:
            print("Iteration:", '%04d' % (iteration+1),"cost=","{:.9f}".format(avg_cost))

    print("tunning completed")

    # test the model
    predictions = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    #to view data: tensorboard --logdir=\logs