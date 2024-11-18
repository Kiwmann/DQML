from .Circuitblocks import Ising,Convolution,Pooling,Convolution_Pooling_CC,Convolution_Pooling_NC,Convolution_Pooling_Full,RandParam
import pennylane as qml
from pennylane import numpy as np

dev1=qml.device('default.qubit',wires=4)
@qml.qnode(dev1)
def circuit_4(scheme,depth,weights,data_in):
    for i in range(2):
        Ising(data_in[i*4:(i+1)*4],list(range(4)),1)

    Convolution_Pooling_Full(weights[0:(depth+2)*4],[0,1,2,3],depth)
    Convolution_Pooling_Full(weights[(depth+2)*4:(depth+2)*6],[1,3],depth)

    return qml.expval(qml.PauliZ(3))

dev2=qml.device('default.qubit',wires=8)
@qml.qnode(dev2)
def circuit_44(scheme,depth,weights,data_in):
    Ising(data_in[:4],[0,1,2,3],1)
    Ising(data_in[4:],[4,5,6,7],1)          

    if scheme=='NCDQML':
        Convolution_Pooling_NC(weights[:(depth+2)*8],[0,1,2,3],[4,5,6,7],depth)
        Convolution_Pooling_NC(weights[(depth+2)*8:(depth+2)*12],[1,3],[5,7],depth)
    if scheme=='CCDQML':
        Convolution_Pooling_CC(weights[:2*(depth+2)*4],[0,1,2,3],[4,5,6,7],depth)
        Convolution_Pooling_CC(weights[(depth+2)*8:(depth+2)*12],[1,3],[5,7],depth)

    return qml.probs(wires=[3,7])

dev3=qml.device('default.qubit',wires=8)
@qml.qnode(dev3)
def circuit_8(scheme,depth,weights,data_in):
    Ising(data_in,list(range(8)),1)
    Convolution_Pooling_Full(weights[:(depth+2)*8],list(range(8)),depth)
    Convolution_Pooling_Full(weights[(depth+2)*8:(depth+2)*12],[1,3,5,7],depth)
    if scheme == 'QCDQML_1m' :
        Convolution_Pooling_Full(weights[(depth+2)*12:(depth+2)*14],[3,7],depth)
        return qml.expval(qml.PauliZ(7))
    elif scheme == 'QCDQML_biased':
        Convolution_Pooling_Full(weights[(depth+2)*12:(depth+2)*14],[3,7],depth)
        return qml.probs(wires=[7])
    elif scheme == 'QCDQML':
        return qml.probs(wires=[3,7])