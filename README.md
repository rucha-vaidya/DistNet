

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

Different Types of Distributed Tensorflow Implemented:


1. Synchronous Implementation
Single Parameter Server which can handle multiple workers parallely but in a synchronous fashion. It waits for all workers to finish an iteration before sending the parameter values after applying all the gradients received from all workers.

2. Asynchronous Implementation - Type 1 
Single Parameter Server which can hadle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it applies the gradients and then sends the parameter values back to the worker without waiting for other workers.

3. Asynchronous Implementation - Type 2 
Single Parameter Server wich can handle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it saves the gradients to apply but immidietly sends the most recent parameter values back to the worker. It doesn't wait to apply the gradients. It also doesn't wait for other workers as well. 

