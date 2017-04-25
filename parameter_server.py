import numpy as np
import json as js
import socket
import cPickle as pickle
import base64 as b64
import zlib as zl
import sys
import threading

TCP_IP = '127.0.0.1'
TCP_PORT = 5008
BUFFER_SIZE = 20  # Normally 1024, but we want fast response
MAX_NUMBER_WORKERS = 2
current_worker_count = 0
lock_worker_count = threading.Lock()
lock_sum_gradients = threading.Lock()
global_sum = []
                         

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
    size = safeReceive(45,conn)
    #size = conn.recv(1024)
    size = pickle.loads(size)
    print("Received the size of gradient ", size)
    data = safeReceive(size,conn)
    local_worker_gradients = pickle.loads(data)
    lock_worker_count.acquire()
    worker_id = 0
    try:
        current_worker_count = current_worker_count+1
        worker_id = current_worker_count    
            
    finally:
        lock_worker_count.release()

    lock_sum_gradients.acquire()
    try:
        if(worker_id == 1):
            global_sum = local_worker_gradients
        else:
            global_sum = global_sum + local_worker_gradients
    finally:
        lock_sum_gradients.release()

    
    print("Received the gradients ", sys.getsizeof(data))
    conn.send(data)
    conn.close()

                 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)


while 1:
    conn, addr = s.accept()
    print 'Connection address:', addr
    threading.Thread(target=handleClient, args=(conn,addr)).start()
   


s.close()
