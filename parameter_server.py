import numpy as np
import json as js
import socket
import pickle
import base64 as b64
import zlib as zl
import sys
from multiprocessing import Process, Queue, Value, Manager
from ctypes import c_char_p

TCP_IP = '127.0.0.1'
TCP_PORT = 5014
BUFFER_SIZE = 20  # Normally 1024, but we want fast response
MAX_NUMBER_WORKERS = 2
ZERO = 0

def add_local_gradients(global_sum, local_gradients):
    for i,grad in enumerate(local_gradients):
        global_sum[i] += grad

def average_gradients(global_sum):
    global_avg = global_sum
    #print("length of average is ,",len(global_avg))
    #print(global_avg[9])
    for i,grad in enumerate(global_sum):
        global_avg[i] = grad / MAX_NUMBER_WORKERS
    #print(global_avg[9])
    return global_avg

def zero_gradients(global_sum):
    for i, grad in enumerate(global_sum):
        global_sum[i].fill(0)

def safeReceive(size,client_socket,port):
    received_size = 0
    data = ''
    temp = ''
    while 1:
        try:
            temp = client_socket.recv(1024)
            data += temp
            received_size = sys.getsizeof(data)
            print(port, ": Received: ", received_size,sys.getsizeof(data),sys.getsizeof(temp))

            if(received_size >= size):
                break
        except:
            print 'Error'

    return data
 

def handleWorker(port,gradients_q,done_flag,global_avg,ack_q):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to port : ", port)
    s.bind((TCP_IP, port))
    s.listen(1)

    while 1:
        conn, addr = s.accept()
        print 'Connection address:', addr
            
        size = safeReceive(45,conn,port)
        size = pickle.loads(size)
        print("Received the size of gradient ", size)
        data = safeReceive(size,conn,port)
        print("Got the data")
        local_worker_gradients = pickle.loads(data)
        print("Sending grad of length ", len(local_worker_gradients) , " to queue")
        gradients_q.put(local_worker_gradients)
        while(done_flag.value == 0):
            pass
        conn.sendall(global_avg.value)
        conn.close()
        ack_q.put(1)
    s.close()

def aggregateSum(gradients_q,done_flag, global_avg,ack_q):
    print("Aggregating process started")
    while(1):
        global_sum = []
            
        for i in xrange(MAX_NUMBER_WORKERS):
            local_worker_gradients = gradients_q.get()
            print("got gradient ", i)
        
            if(i == 0):
                global_sum = local_worker_gradients
            else:
                add_local_gradients(global_sum, local_worker_gradients) 

        print("Got all gradients, averaging them")
        avg = average_gradients(global_sum)
        global_avg.value = pickle.dumps(avg, pickle.HIGHEST_PROTOCOL)
        done_flag.value = 1
        for i in xrange(MAX_NUMBER_WORKERS):
            val = ack_q.get()
        done_flag.value = 0
        print("Iteration complete")
            


def main(argv=None):
    if(len(sys.argv) != 3):
        print("Port number and number of workers required")
        sys.exit()
    global MAX_NUMBER_WORKERS
    MAX_NUMBER_WORKERS = int(sys.argv[2])
    manager = Manager()
    global_avg = manager.Value(c_char_p, "")
    done_flag = manager.Value('i', 0)

    gradients_q = Queue()
    ack_q = Queue()

    master_process = Process(target=aggregateSum, args=(gradients_q,done_flag, global_avg, ack_q))
    master_process.start()
    port = int(sys.argv[1])
    
    for i in xrange(MAX_NUMBER_WORKERS):
        process_port = port + i
        p = Process(target=handleWorker, args=(process_port,gradients_q,done_flag,global_avg, ack_q))
        p.start()

    while(1):
        pass

if __name__ == "__main__":
        main(sys.argv)

