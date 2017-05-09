

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

Different Types of Distributed Tensorflow Implemented:


1. Synchronous Implementation
Single Parameter Server which can handle multiple workers parallely but in a synchronous fashion. It waits for all workers to finish an iteration before sending the parameter values after applying all the gradients received from all workers.

2. Asynchronous Implementation - Apply gradients and reply
Single Parameter Server which can hadle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it applies the gradients and then sends the parameter values back to the worker without waiting for other workers.

3. Asynchronous Implementation - Reply without applying gradients 
Single Parameter Server wich can handle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it saves the gradients to apply but immidietly sends the most recent parameter values back to the worker. It doesn't wait to apply the gradients. It also doesn't wait for other workers as well. 

4. Asynchronous Implementation - Like Project Adam
Single Parameter Server wich can handle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it saves the gradients to apply but immidietly sends the most recent parameter values back to the worker. It does this in a lock free fashion. There is no lock acquire/release delay. No waiting for other workers as well.

5. Asynchronous Implementation - Multiple Parameter Servers
Multiple (2) Parameter Servers which distribute the parameter values among themselves. It works in an asynchronous fashion. Worker sends the respective gradients to the parameter servers and wait for replies. The parameter servers send the latest gradients without lock delay back to the worker. No waiting for other workers as well.
