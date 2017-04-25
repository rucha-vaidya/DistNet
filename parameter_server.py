import numpy as np
import json as js
import socket
import cPickle as pickle
import base64 as b64
import zlib as zl
import sys


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
    
TCP_IP = '127.0.0.1'
TCP_PORT = 5008
BUFFER_SIZE = 20  # Normally 1024, but we want fast response
                                          
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

while 1:
    conn, addr = s.accept()
    print 'Connection address:', addr
    data = ''
    temp = ''

    size = safeReceive(45,conn)
    size = pickle.loads(size)
    print("Received the size of gradient ", size)
    data = safeReceive(size,conn)
    print("Received the gradients ", sys.getsizeof(data))
    conn.send(data)
    conn.close()


s.close()
