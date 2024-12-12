import pennylane as qml
from pennylane import numpy as np
import qutip

# Ising Block used in Embedding and Convolutional layer
# Hadamard and RZ in each qubit and ZZ with neighboring qubit
# number of qubit = number of parameter
# th_i for ith  RZ
# (1/2)(pi-th_1)(pi-th_2) in ZZ
def Ising(weights,wires,depth): #size of circuit = number of parameter
    size=len(wires)
    if size==2: #for 2 qubit Ising, only one ZZ interaction is required.
        for d in range(depth):
            [qml.Hadamard(wires=w) for w in wires]
            [qml.RZ(weights[size*d+i],wires=wires[i]) for i in range(size)]
            qml.IsingZZ(0.5*(np.pi-weights[d*size])*(np.pi-weights[d*size+1]),wires=[wires[0],wires[1]])
    else:
        for d in range(depth):
            [qml.Hadamard(wires=w) for w in wires]
            [qml.RZ(weights[size*d+i],wires=wires[i]) for i in range(size)]
            [qml.IsingZZ(0.5*(np.pi-weights[size*d+i])*(np.pi-weights[size*d+(i+1)%size]),wires=[wires[i],wires[(i+1)%size]]) for i in range(size)]

def Convolution(weights,wires,depth): 
    size=len(wires)
    for d in range(depth):
        [qml.Hadamard(wires=w) for w in wires]
        for i in range(int(size/2)):
            qml.CZ(wires=[wires[i*2],wires[(i*2+1)%size]])
        [qml.RX(weights[size*d+i],wires=wires[i]) for i in range(size)]

        if size!=2: 
            [qml.Hadamard(wires=w) for w in wires]
            for i in range(int(size/2)):
                qml.CZ(wires=[wires[i*2+1],wires[(i*2+2)%size]])
            [qml.RX(weights[size*d+i],wires=wires[i]) for i in range(size)]



# Pooling Circuit
def Pooling(weights,m_qubit,t_qubit):
    m_f=qml.measure(m_qubit)
    qml.cond(m_f==0,qml.RZ)(weights[0],wires=t_qubit)
    qml.cond(m_f==0,qml.RX)(weights[1],wires=t_qubit)
    qml.cond(m_f==1,qml.RZ)(weights[2],wires=t_qubit)
    qml.cond(m_f==1,qml.RX)(weights[3],wires=t_qubit)

def Convolution_Pooling_CC(weights,wires_1,wires_2,depth):
        qml.Barrier(wires=wires_1,only_visual=True)
        qml.Barrier(wires=wires_2,only_visual=True)
        size=len(wires_1)
        wires=[wires_1,wires_2]
        for i in range(2):
            Convolution(weights[i*depth*size:(i+1)*depth*size],wires[i],depth)
        qml.Barrier(wires=wires_1,only_visual=True)
        qml.Barrier(wires=wires_2,only_visual=True)
        poolingweights=weights[2*depth*size:]
        for j in range(int(size/2)):
            Pooling(poolingweights[8*j:8*j+4],wires_1[j*2],wires_2[j*2+1])
            Pooling(poolingweights[8*j+4:8*j+8],wires_2[j*2],wires_1[j*2+1])

    #total parameters: 2*(depth+2)*(size per module)

def Convolution_Pooling_NC(weights,wires_1,wires_2,depth):
        qml.Barrier(wires=wires_1,only_visual=True)
        qml.Barrier(wires=wires_2,only_visual=True)
        size=len(wires_1)
        wires=[wires_1,wires_2]
        for i in range(2):
            Convolution(weights[i*depth*size:(i+1)*depth*size],wires[i],depth)
        poolingweights=weights[2*depth*size:]
        qml.Barrier(wires=wires_1,only_visual=True)
        qml.Barrier(wires=wires_2,only_visual=True)        
        for j in range(int(size/2)):
            Pooling(poolingweights[8*j:8*j+4],wires_1[j*2],wires_1[j*2+1])
            Pooling(poolingweights[8*j+4:8*j+8],wires_2[j*2],wires_2[j*2+1])
    #total parameters: 2*(depth+2)*(size per module)

def Convolution_Pooling_Full(weights,wires,depth):
    qml.Barrier(wires=wires,only_visual=True)
    size=len(wires)
    Convolution(weights[:depth*size],wires,depth)
    qml.Barrier(wires=wires,only_visual=True)
    poolingweights=weights[depth*size:]
    for j in range(int(size/2)):
       Pooling(poolingweights[4*j:4*j+4],wires[j*2],wires[j*2+1])
    #total parameters: (depth+2)*size
    

def RandParam(num_para):
    P=[]
    for i in range(num_para):
        P.append(2*np.pi*np.random.rand())
    return np.array(P)

def Haar_gate(w):
    Haar_Unitary = np.array(qutip.random_objects.rand_unitary_haar(2**len(w)))
    qml.QubitUnitary(Haar_Unitary,wires=w)
    