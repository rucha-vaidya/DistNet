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
from multiprocessing import Process, Queue, Value, Manager, Lock
from ctypes import c_char_p 

from datetime import datetime
import time

import tensorflow as tf

import cifar10

TCP_IP = '127.0.0.1'
s = 0
MAX_WORKERS = 0
lock = None

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100002,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1000,
                            """How often to log results to the console.""")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)

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

def handleWorker(port,gradients_q,global_var_vals):
    global lock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to port : ", port)
    s.bind((TCP_IP, port))
    s.listen(1)

    while 1:
        conn, addr = s.accept()
        #print('Connection address:', addr)
            
        size = safe_recv(8,conn)
        size = pickle.loads(size)
        data = safe_recv(size,conn)
        #print("Received size: ", size)
        local_worker_gradients = pickle.loads(data)
        gradients_q.put(local_worker_gradients)
        size = len(global_var_vals.value)
        #print("Global var val size: ", size)
        size = pickle.dumps(size, pickle.HIGHEST_PROTOCOL)
        conn.sendall(size)
        global lock
        lock.acquire()
        send_data = global_var_vals.value
        lock.release()
        conn.sendall(send_data)
        conn.close()
    s.close()


def train():
  """Train CIFAR-10 for a number of steps."""

  g1 = tf.Graph()
  with g1.as_default():
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #global_step = tf.contrib.framework.get_or_create_global_step()

    
    global_step = tf.Variable(-1, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step+1)
    
    cifar10.build_graph()
       
    placeholder_gradients = []

    #with tf.device("/gpu:0"):
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        placeholder_gradients.append((tf.placeholder('float', shape=var.get_shape()) ,var))
    feed_dict = {}
       
    i=0
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
       feed_dict[placeholder_gradients[i][0]] = np.zeros(placeholder_gradients[i][0].shape)
       i=i+1
    train_op = cifar10.train_part2(global_step,placeholder_gradients)
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d,(%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, 
              examples_per_sec, sec_per_batch))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:

      for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
          print(v)
      # Sending the initial value of variables
      var_val = []
      for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        var_val.append(mon_sess.run(v, feed_dict=feed_dict))
      send_data = pickle.dumps(var_val, pickle.HIGHEST_PROTOCOL)
      global global_var_vals
      global lock
      lock.acquire()
      global_var_vals.value = send_data
      lock.release()
      size = len(send_data)
      size = pickle.dumps(size, pickle.HIGHEST_PROTOCOL)
      for i in xrange(MAX_WORKERS):
        conn, addr = s.accept()
        conn.sendall(size)
        conn.sendall(send_data)
        conn.close()
      print("Sent initial var values to workers")
      while not mon_sess.should_stop():
        val = mon_sess.run(global_step, feed_dict=feed_dict)
        #print("Iteration: ", val)
        if(val == (FLAGS.max_steps - 1)):
            print("Global step val while stoping.")
            sys.exit()
        recv_grads = gradients_q.get()
        #print("received gradients from worker")
        feed_dict = {}
        for i,grad_var in enumerate(recv_grads): 
           feed_dict[placeholder_gradients[i][0]] = recv_grads[i]

        res = mon_sess.run(train_op, feed_dict=feed_dict)
        var_val = []
        #print("Run complete with new values")
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            var_val.append(mon_sess.run(v, feed_dict=feed_dict))
        global global_var_vals
        global lock
        lock.acquire()
        global_var_vals.value = pickle.dumps(var_val, pickle.HIGHEST_PROTOCOL)
        lock.release()
        #print("New values of variables sent ")


def main(argv=None):  # pylint: disable=unused-argument
  if(len(sys.argv) != 3):
      print("Port number and no of workers required")
      sys.exit()
  global s
  global port
  global MAX_WORKERS
  port = int(sys.argv[1])
  MAX_WORKERS = int(sys.argv[2])
  global gradients_q
  global global_var_vals
  global lock
  gradients_q = Queue()
  manager = Manager()
  global_var_vals = manager.Value(c_char_p, "")
  lock = Lock()

  for i in xrange(MAX_WORKERS):
        process_port = port + i + 1
        p = Process(target=handleWorker, args=(process_port,gradients_q,global_var_vals))
        p.daemon=True
        p.start()

  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  total_start_time = time.time()
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  print("Connecting to port : ", port, " and no of workers: ", MAX_WORKERS)
  s.bind((TCP_IP, port))
  s.listen(1)
  train()
  print("--- %s seconds ---" % (time.time() - total_start_time))

if __name__ == '__main__':
  tf.app.run()
