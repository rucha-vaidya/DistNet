
DistNet : Distributed Training for DNN 

Dataset Used:
CIFAR-10 is a common benchmark in machine learning for image recognition.
http://www.cs.toronto.edu/~kriz/cifar.html

DNN used:
Alexnet

Code in this directory contains code training a convolutional neural network (CNN) in a Distributed fashion. It uses single GPU TensorFlow code as a base and builds it up for it to work in a distributed fashion. It achieves an accuracy of 86.01% but in a shorter amount of time.

Different Types of Distributed Tensorflow Implemented:


1. Synchronous Implementation
Single Parameter Server which can handle multiple workers parallely but in a synchronous fashion. It waits for all workers to finish an iteration before sending the parameter values after applying all the gradients received from all workers.
Files: cifar10_train_sync.py  ps_sync.py 
Output Files: ps_sync_output worker_sync_1 worker_sync_2

2. Asynchronous Implementation - Apply gradients and reply
Single Parameter Server which can hadle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it applies the gradients and then sends the parameter values back to the worker without waiting for other workers.
Files: ps_async_mp_apply_send.py   
Output Files: ps_async_apply_output worker_1_async_apply_send_output worker_2_async_apply_send_output

3. Asynchronous Implementation - Reply without applying gradients 
Single Parameter Server wich can handle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it saves the gradients to apply but immidietly sends the most recent parameter values back to the worker. It doesn't wait to apply the gradients. It also doesn't wait for other workers as well. It works in a locked fashion.
Files: ps_async_locked.py . 
Output Files: ps_async_locked_output worker_1_async_locked_output worker_2_async_locked_output


4. Asynchronous Implementation - Like Project Adam
Single Parameter Server wich can handle multiple workers parallely but in an asynchronous fashion. When it receives gradient values from a worker, it saves the gradients to apply but immidietly sends the most recent parameter values back to the worker. It does this in a lock free fashion. There is no lock acquire/release delay. No waiting for other workers as well.
Files: cifar10_train_async_mp.py ps_async_mp.py . 


5. Asynchronous Implementation - Multiple Parameter Servers
Multiple (2) Parameter Servers which distribute the parameter values among themselves. It works in an asynchronous fashion. Worker sends the respective gradients to the parameter servers and wait for replies. The parameter servers send the latest gradients without lock delay back to the worker. No waiting for other workers as well.
Files: cifar10_train_multips_async.py multi_ps_async_part1.py multi_ps_async_part2.py . 
Output Files: multips_1_output multips_2_output multips_worker1_output multips_worker2_output . 
