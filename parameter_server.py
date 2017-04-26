import numpy as np
import json as js
import socket
import cPickle as pickle
import base64 as b64
import zlib as zl
import sys
import threading

TCP_IP = '127.0.0.1'
TCP_PORT = 5009
BUFFER_SIZE = 20  # Normally 1024, but we want fast response
MAX_NUMBER_WORKERS = 1
current_worker_count = 0
ZERO = 0
lock_worker_count = threading.Lock()
lock_sum_gradients = threading.Lock()
ready_to_send = False
global_sum = []
global_avg = []
#global_sum = np.array(global_sum)
#global_avg = np.array(global_avg)

def add_local_gradients(local_gradients):
    for i,grad in enumerate(local_gradients):
        global_sum[i] += grad

def average_gradients():
    global_avg = global_sum
    print("length of average is ,",len(global_avg))
    print(global_avg[9])
    for i,grad in enumerate(global_sum):
        global_avg[i] = grad / MAX_NUMBER_WORKERS
    print(global_avg[9])

def zero_gradients():
    for i, grad in enumerate(global_sum):
        global_sum[i] = global_sum[i].fill(ZERO)

def safeReceive(size,client_socket):
    received_size = 0
    data = ''
    temp = ''
    while 1:
        try:
            temp = client_socket.recv(1024)
            data += temp
            received_size = sys.getsizeof(data)
            print("Received: ", received_size,sys.getsizeof(data),sys.getsizeof(temp))

            if(received_size >= size):
                break
        except:
            print 'Error'

    return data
  
def handleClient(client_socket, client_address):
    global current_worker_count
    global ready_to_send
    global global_sum
    global global_avg
    size = safeReceive(45,conn)
    #size = conn.recv(1024)
    size = pickle.loads(size)
    print("Received the size of gradient ", size)
    data = safeReceive(size,conn)
    print("Got the data")
    local_worker_gradients = pickle.loads(data)
    print(len(local_worker_gradients))  

    lock_sum_gradients.acquire()
    print("acquired sum gradients lock")
    try:
        worker_id = current_worker_count
        print("correct worker id, ",worker_id)
        if(worker_id == 0):
            global_sum = local_worker_gradients
        else:
            add_local_gradients(local_worker_gradients)
        
        print("length of global sum ",len(global_sum))  
        print(global_sum[9])
        
        current_worker_count = current_worker_count+1
        if(worker_id == MAX_NUMBER_WORKERS - 1):
            print("calculating average by, ",worker_id)
            average_gradients()
            ready_to_send = True
            zero_gradients()

    finally:
        lock_sum_gradients.release()
        print("lock released")

 
    while(ready_to_send == False):
        pass

    client_socket.send(pickle.dumps(global_avg,pickle.HIGHEST_PROTOCOL))
    
    data = pickle.dumps(global_avg,pickle.HIGHEST_PROTOCOL)
    print("length is, ",len(pickle.loads(data)))
    lock_sum_gradients.acquire()
    print("lock acquired again")
    try:
        current_worker_count = current_worker_count-1
        worker_id = current_worker_count
        if(worker_id == 0):
            ready_to_send = False

    finally:
        lock_sum_gradients.release()
        print("lock released again")
    
    client_socket.close()



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

while 1:
    conn, addr = s.accept()
    print 'Connection address:', addr
    threading.Thread(target=handleClient, args=(conn,addr)).start()
   


s.close()
