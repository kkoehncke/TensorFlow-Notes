#Up and Running with TensorFlow
##Managing Graphs
Initializing variables and defining a function
```
x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name= 'y')
f = x*x*y + y + 2
```

One way of running TF session
```
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print (result)
>>> 42
sess.close()
```

Globally creates all variable nodes without having to specify sess.run(...), 
Automatically closing as well
```
init = tf.global_variables_initializer()  # prepare an init node

with tf.Session() as sess:
    init.run()  # actually initialize all the variables
    result = f.eval()
```

We can also create an interactive session, which tells TensorFlow to set the interactive session as default; still have to close session:
```
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
>>> 42
sess.close()
```

TensorFlow program typically broken into two parts:

	- Graph Construction Phase 
	- Execution Phase

Any node you create is automatically added to the default graph:
```
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()
>> True
```

In most cases this is fine, but sometimes you may want to manage multiple independent graphs. 
You can do this by creating a new Graph and temporarily making it the default graph inside a with block, like so:

```
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
>> True
x2.graph is tf.get_default_graph()
>> False
```

**Run tf.reset_default_graph() to prevent a graph having duplicate nodes when running jupyter cells multiple times**

Tensorflow automatically determines the set of nodes needed and evaluates those nodes first:
```
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15
```

- Tensorflow looks at program and sees we want to evaluate y
  - y depends on x which depends w so we evaluate: w -> x -> y
- Next we want to evaluate z 
  - z depends on x which depends on w so we evaluate: w -> x -> z
- We note that in each separate evaluation, **TensorFlow does not store result of previous evaluations**
  - All node values are dropped between each evaluation 

If we want an efficient evaluation we can do the following:
```
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15
```

##Linear Regression 

Here is an example of computing normal equation for sample housing data:
```
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
```

##Batch Gradient Descent 

**Before performing, remember to normalize input variables using something like StandardScaler()**

Notes:
- random_uniform() (acts like Numpy rand() function), generates random values
- assign() performs assignment of variable with new value 

```
# Construction Phase
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

#Execution Phase, loops over number of iterations chosen
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
```

We can however take advantage of TensorFlow's *autodiff* feature so we don't need to know the gradient symbolically:

```
gradient = tf.gradients(mse, [theta])[0]
```

We can also use a built-in optimizer (won't need to use gradient term as its already baked in):

```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```

So our example turns into:
```
# Construction Phase
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# Replaced HERE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Execution Phase, loops over number of iterations chosen
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
```

##Placeholder Nodes

Placeholder Nodes just output data you tell it to output at runtime:
```
# We can specify the shape of our placeholder
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
        B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
 
print(B_val_1)
>>> [[ 6.  7.  8.]]
print(B_val_2)
>>> [[  9.  10.  11.]
    [ 12.  13.  14.]]
```

##Mini Batch Gradient Descent 
With our placeholders, we can implement mini-batch gradient descent as follows:
```
# Construction Phase
n_epochs = 10
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

#Replaced with placeholders
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

#Obtain our batch of a given size
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)  
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

# Execution Phase, loops over number of iterations chosen
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
       for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
```
##Saving Model
We can save our trained model for later usage using a *Saver* node as follows:
```
#Creates a Saver node
saver = tf.train.Saver() 
#Creates a Saver node that specifies what variables to save / restore
saver = tf.train.Saver({"weights": theta})
...
with tf.Session() as sess:
	#Saves model to path name
	save_path = saver.save(sess, "PATH_NAME")
	
	#Restores model; do not call sess.run(init)
	saver.restore(sess, "PATH_NAME")
```

By default, the *Saver* node saves the graph structure to a *.meta* file. Hence, after we train our model, a brand new invocation of our model would look something like this:
```
reset_graph()

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure**
theta = tf.get_default_graph().get_tensor_by_name("theta:0") 

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval()  
```

##Tensorboard
Instead of using print statements to track our progress during training, we can utilize Tensorboard.

First we need to write information about our graph and statistics associated with our model (e.g. MSE) to a log file that Tensorboard can read. We need to create a new log file everytime otherwise Tensorboard will aggregate the results; we do this by adding a timestamp to the log as follows:

```
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
```

We then can write to our log file during the construction phase as follows:

```
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```

During our mini-batch training, we want to evaluate the MSE during each mini-batch:
```
    for batch_index in range(n_batches):
        X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
        if batch_index % 10 == 0:
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
            step = epoch * n_batches + batch_index
            file_writer.add_summary(summary_str, step)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
```

**Remember to close file writer at end of program**
```
file_writer.close()
```

##TensorFlow Semantics
###Name Scopes 
We can create *name scopes* to group together related nodes. For example:
```
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
```

We see that now each op defined within our name scope has the prefix loss/:
```
print error.op.name
>>> loss/sb
print mse.op.name
>>> loss/mse
```

###Modularity 
We can define functions as normal to perform certain functionality modulary:

```
#Define ReLu node
def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```

![](http://i.markdownnotes.com/image_pROn5Ug.jpg)

In combination with a name scope, we can make our graph look a lot cleaner:
```
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                         
        w = tf.Variable(tf.random_normal(w_shape), name="weights")    
        b = tf.Variable(0.0, name="bias")                             
        z = tf.add(tf.matmul(X, w), b, name="z")                      
        return tf.maximum(z, 0., name="max")    
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```

![](http://i.markdownnotes.com/image_pJh8aMp.jpg)

###Sharing Variables
There multiple ways within TensorFlow to share variables. 

The standard way of defining a variable and passing it as a parameter to a function still works as we see with our ReLu example:

```
def relu(X, threshold):
    with tf.name_scope("relu"):
        [...]
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
[...]
```

This method can be cumbersome if we have multiple parameters. Another way is to set the shared variable as an attribute in the function itself:

```
def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        [...]
        return tf.maximum(z, relu.threshold, name="max")
```

A slightly more complicated but more compact way of sharing variables is by using TensorFlow's *get_variable().* We can use it as follows:

```
def relu(X):
	//Creates variable if not defined, reuses for later use
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    [...]
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")
```

![](http://i.markdownnotes.com/image_FIKMujA.jpg)
#Introduction To Neural Networks
##Training a Simple DNN
TensorFlow has many pre-built neural networks within its high-level API *TF.Learn*. However, we might want more control over the architecture of our neural network, so we will implement a simple DNN using TensorFlow and the MNIST dataset.

###Construction Phase
We first specify the number of input and output nodes, as well how many hidden layers we want (and the number of nodes in each layer):

```
import tensorflow as tf
#In our case, we want 2 hidden layers
n_inputs = 28*28  # MNIST, each image is 28x28 pixels 
n_hidden1 = 300   
n_hidden2 = 100
n_outputs = 10
```

Next we define our placeholder nodes to hold our training data and target:

```
//We know that X will be a matrix with each column being a feature,
//hence why we specify the y-axis to be size of n_inputs
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
```

Now we want to create a function which will create a neuron layer, whether it is a hidden or output layer does not matter as we can pass parameters to our function to define specifically what kind of layer it is :

```
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
		//We calculate stddev this way because we want to initialize W 
		//randomly using a truncated Gaussian distribution
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
```

Now we are ready to build our neural network:

```
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
```

(TensorFlow already has functions that do all this for you like *tf.layers.dense()* which creates a fully connected layer):

```
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
```

Now that we have the architecture of our neural network set up, we need to define our cost function that we use to train our model. We will use cross entropy seeing as we are trying to classify the different digits in the MNIST data set. 

*sparse_softmax_cross_entropy_with_logits()* is equivalent to applying the softmax activation function and then computing the cross entropy, but it is more efficient, and it properly takes care of corner cases like logits equal to 0

```
with tf.name_scope("loss"):	
	//Computes cross entropy with logits i.e. output before going through softmax
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    //Compute mean cross entropy 
	loss = tf.reduce_mean(xentropy, name="loss")
	//Computes log loss
	loss_summary = tf.summary.scalar('log_loss', loss)
```

Now all we need to do is define our *GradientDescentOptimizer* and how we evaluate our model; in this case we will simply use accuracy:

```
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
	//Checks whether highest logit corresponds to target class
	//Returns vector of boolean values
    correct = tf.nn.in_top_k(logits, y, 1)
	//Need to cast booleans to floats to compute average
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```

Lastly, we call our global variable initializer and define our Saver and that concludes the construction phase!:

```
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```

###Execution Phase

We begin by loading in the MNIST dataset. TensorFlow has its own helper function that gets the data, scales it between 0 and 1, shuffles it, and provides a function to load one mini batch at a time:

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
```

Now we can train our model as usual:

```
n_epochs = 40
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```


And test our model as usual:

```
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
	
print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.test.labels[:20])
>>> Predicted classes: [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
>>> Actual classes:    [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
```


##Fine Tuning Hyperparameters
We need to know the optimal combination of hyperparameter settings that will maximize the performance on the task at hand. For neural networks, there are a multitude of hyperparameters to be tuned. We will examine the most important ones individually.
###Number of Hidden Layers
For most problems, using 1-2 hidden layers will work fine for most problems. For more complex problems, gradually increase number of hidden layers until overfitting starts to take effect.
###Number of Neurons Per Layer
Number of neurons in input layer = number of features in input data
Number of neurons in output layer = number of output values

Number of neurons in hidden layer is a black art, try different numbers applied across all layers (50, 100, 300, etc.)
###Activation Functions
Number of different kinds of activation functions (sigmoid, tanh, ReLu, etc.). In most cases, ReLu is good default option. 

Use softmax for multi-class classification, use sigmoid for binary classification

#Training Deep Neural Networks
##Vanishing & Exploding Gradients
Vanishing - Gradient gets smaller and smaller, leaving lower layer connection weights virtually unchanged during update phase, leading to a non-convergent solution 

Exploding - Primarily occurs in RNNs, gradient gets larger and larger, causing weights to increase rapidly, leading to a non-convergent solution 

We can alleviate this problem one way using different kinds of initialization techniques on the weights as shown in the table below:

![](http://i.markdownnotes.com/image_26WNkZE.jpg)

*Xavier Initialization* is the name for the Logistic function
*He Initialization* is the name for the ReLU function

By default, TensorFlow's *tf.layers.dense()* uses Xavier initialization. To change to He initialization we do the following:

```
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name="hidden1"
```


##Non-Saturating Activation Functions
In deep learning, ReLU is used often because it does not saturate for positive values and is computed quickly. 

One problem with the ReLU known as *dying ReLUs*: During training some neurons effectively die; only output 0. During training, if a neuron’s weights get updated such that the weighted sum of the neuron’s inputs is negative, it will start outputting 0. When this happens, the neuron is unlikely to come back to life since the gradient of the ReLU function is 0 when its input is negative.

Solution: **leaky ReLU**: $$LeakyReLU_α(z) = max(αz, z)$$

