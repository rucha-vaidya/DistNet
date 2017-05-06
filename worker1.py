# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import cPickle as pickle
import socket

from datetime import datetime
import time

import tensorflow as tf

import cifar10

TCP_IP = '127.0.0.1'
TCP_PORT = 5014

port = 0
s = 0

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
def safe_recv(size, server_socket):
  data = ''
  temp = ''
  recv_size = 0
  while 1:
    try:
        temp = server_socket.recv(size-len(data))
        data += temp
        recv_size = len(data)
        if recv_size >= size:
            break
    except:
        print("Error")
  return data

def train():
  """Train CIFAR-10 for a number of steps."""

  g1 = tf.Graph()
  with g1.as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs2()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    grads  = cifar10.train_part1(loss, global_step)

    only_gradients = [g for g,_ in grads]
    only_vars = [v for _,v in grads]

       
    placeholder_gradients = []

    #with tf.device("/gpu:0"):
    for grad_var in grads :
        placeholder_gradients.append((tf.placeholder('float', shape=grad_var[0].get_shape()) ,grad_var[1]))

 
    
    feed_dict = {}
       
    for i,grad_var in enumerate(grads): 
       feed_dict[placeholder_gradients[i][0]] = np.zeros(placeholder_gradients[i][0].shape)
      
  
  
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train_part2(global_step,placeholder_gradients)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.
        

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:

      global port
      while not mon_sess.should_stop():
        
        gradients = mon_sess.run(only_gradients,feed_dict = feed_dict)
        # pickling the gradients
        send_data = pickle.dumps(gradients,pickle.HIGHEST_PROTOCOL)
        # finding size of pickled gradients
        to_send_size = len(send_data)
        # Sending the size of the gradients first
        send_size = pickle.dumps(to_send_size, pickle.HIGHEST_PROTOCOL)
        s.sendall(send_size)
        # sending the gradients
        s.sendall(send_data)
        recv_size = safe_recv(8, s)
        recv_size = pickle.loads(recv_size)
        recv_data = safe_recv(recv_size, s)
        gradients2 = pickle.loads(recv_data)
        #print("Recevied gradients of size: ", len(recv_data))
        feed_dict = {}
       

        for i,grad_var in enumerate(gradients2): 
           feed_dict[placeholder_gradients[i][0]] = gradients2[i]
           #print(gradients[i].shape)
           #print(gradients2[i].shape)


        res = mon_sess.run(train_op, feed_dict=feed_dict)

def main(argv=None):  # pylint: disable=unused-argument
  global port
  global s
  if(len(sys.argv) != 3):
      print("<port>, <worker_id> required")
      sys.exit()
  port = int(sys.argv[2]) + int(sys.argv[1]) 
  print("Connecting to port ", port)
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  total_start_time = time.time()
  # Opening the socket and connecting to server
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((TCP_IP, port))
  train()
  s.close()
  print("--- %s seconds ---" % (time.time() - total_start_time))


if __name__ == '__main__':
  tf.app.run()
