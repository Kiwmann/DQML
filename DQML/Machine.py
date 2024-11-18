import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pickle
from functools import partial

from .Circuitblocks import RandParam
from .Loss import accuracy_Expval,cost_squareloss,learnedmeasure_squareloss,biased_onemeasured_squareloss
from .Circuit import circuit_4, circuit_44, circuit_8



def DQML(scheme,depth,iteration,train_validation_set):
    batch_size=512
    num_val=512
    if scheme =='NCDQML' or scheme  == 'CCDQML':
        circuit=lambda x,y :circuit_44(scheme,depth,x,y)
        num_qubits=8

    elif scheme =='non_DQML':
        circuit=lambda x,y :circuit_4(scheme,depth,x,y)
        num_qubits=4

    elif scheme =='QCDQML' or scheme == 'QCDQML_1m' or scheme == 'QCDQML_nonbiased':
        circuit=lambda x,y :circuit_8(scheme,depth,x,y)
        num_qubits=8
    if scheme=='NCDQML' or scheme== 'CCDQML':
        weights=RandParam((depth+2)*12)
    if scheme=='non_DQML':
        weights=RandParam((depth+2)*6)  
    if scheme=='QCDQML_1m' or scheme=='QCDQML_nonbiased':
        weights=RandParam((depth+2)*14)
    if scheme == 'QCDQML':
        weights=RandParam((depth+2)*12)

    accuracy=accuracy_Expval
    cost=learnedmeasure_squareloss
    cost= partial(cost,circuit)
    if scheme == 'QCDQML_1m':
        cost= biased_onemeasured_squareloss
        cost=partial(cost,circuit)
    if scheme=='QCDQML_nonbiased' or scheme=='non_DQML':
        cost=cost_squareloss
        cost=partial(cost,circuit)
    opt=qml.AdamOptimizer()
    start_time=time.time()
    feats_train=train_validation_set[0]
    Y_train=np.array(train_validation_set[1])
    feats_val=train_validation_set[2][:num_val]
    Y_val=train_validation_set[3][:num_val]
    num_train=train_validation_set[4]
    cost_list=[]
    acc_val_list=[]
    bias=np.array([1,-1,-1,1])
    if scheme == 'QCDQML_1m':
        bias=np.array([1,-1])

    print(qml.draw(circuit,decimals=None)(weights,np.random.rand(8)))
    for it in range(iteration):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        if scheme == 'non_DQML' or scheme == 'QCDQML_nonbiased':
            opt_result = opt.step_and_cost(cost, weights,  feats_train_batch, Y_train_batch)
            weights, _, _ = opt_result[0]
            predictions_val = np.sign([circuit(weights, f) for f in feats_val])
            acc_val = accuracy(Y_val, predictions_val)


        elif scheme=='NCDQML' or scheme== 'CCDQML' or scheme== 'QCDQML':
            opt_result = opt.step_and_cost(cost,weights,bias,feats_train_batch, Y_train_batch)
            weights,bias,_,_=opt_result[0]
            predictions_val=[]
            for f in feats_val:
                outcome=circuit(weights,f)
                predictions_val.append(np.sign(bias[0]*outcome[0]+bias[1]*outcome[1]+bias[2]*outcome[2]+bias[3]*outcome[3]))
            acc_val = accuracy(Y_val,predictions_val)
            
        elif scheme=='QCDQML_1m':
            opt_result = opt.step_and_cost(cost,weights,bias,feats_train_batch, Y_train_batch)
            weights,bias,_,_=opt_result[0]
            predictions_val=[]
            for f in feats_val:
                outcome=circuit(weights,f)
                predictions_val.append(np.sign(bias[0]*outcome[0]+bias[1]*outcome[1]))
            acc_val = accuracy(Y_val,predictions_val)

        cost_list.append(opt_result[1])
        acc_val_list.append(acc_val)

        if it%10 ==0:
            print('(',it,') loss:',np.round(opt_result[1],2), 'accuracy:',np.round(acc_val,2))

    print('---final---')
    print('loss:',np.round(opt_result[1],2), 'accuracy:',np.round(acc_val,2))
    print("%s mins" % round((time.time() - start_time)/60,1))
    print('bias:',bias)

    plt.title('depth='+str(depth))
    plt.plot(cost_list,'--o',label='Loss')
    plt.plot(acc_val_list,'--o',label='Accuracy')
    plt.xlabel("iterations")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc=(0.75,1.1))
    plt.ylim([0,1.2])
    plt.show()
    return cost_list, acc_val_list, weights, bias
