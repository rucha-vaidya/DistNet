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
    images, labels = cifar10.distorted_inputs()

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
  
    train_op = cifar10.train_part2(global_step,placeholder_gradients)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feeds = []
    print("Reached here")
    for i,grad_var in enumerate(grads): 
        feeds.append(placeholder_gradients[i][0])
    # Partial Run
    print("Reached here", len(feeds))
    for x in feeds:
        print(x,)
    h = sess.partial_run_setup([only_gradients, train_op], feeds)
    print("Reached here")


    for i in xrange(10):
        res_grads = sess.partial_run(h, only_gradients, feed_dict = feed_dict)

        feed_dict = {}
        for i,grad_var in enumerate(res_grads): 
           feed_dict[placeholder_gradients[i][0]] = res_grads[i]

        res_train_op = sess.partial_run(h, train_op, feed_dict=feed_dict)


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
  '''
  # Opening the socket and connecting to server
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((TCP_IP, port))
  '''
  train()
  '''
  s.close()
  '''
  print("--- %s seconds ---" % (time.time() - total_start_time))


if __name__ == '__main__':
  tf.app.run()
