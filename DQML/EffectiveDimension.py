import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
import time
import qutip
import pickle
from .Circuitblocks import RandParam
from .Circuit import circuit_22_Haar, circuit_44_Haar

def ParityFisherelement(dp):
    odddp=0
    evendp=0
    p_ord=[1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1]
    if len(dp)>=16:
        print('Maximum 4 measurement')
        raise ValueError
    for k in range(len(dp)):
        if p_ord[k]==1:
            evendp+=dp[k]
        else:    
            odddp+=dp[k]
    return evendp,odddp

def Fisher(weights,circuit):
    F=[]
    dp=qml.jacobian(circuit)(weights)
    p_ord=[1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1]
    p= circuit(weights)
    evenp=0
    oddp=0
    for i in range(len(p)):
        if p_ord[i]==1:
            evenp+=p[i]
        else:    
            oddp+=p[i]
    yprob=[evenp,oddp]
    for i in range(len(weights)):
        temp=[]
        for j in range(len(weights)):
            edp,odp=ParityFisherelement(dp[:,i])
            dpi=[edp,odp]
            edp,odp=ParityFisherelement(dp[:,j])
            dpj=[edp,odp]         
            temp.append(np.sum([dpi[i]*dpj[i]/(yprob[i]+1e-15) for i in range(2)]))
        F.append(temp)

    return F



def Fisherinformation(scheme,depth,it,x_sample):
    if scheme in ['NCDQML4','CCDQML4','QCDQML4']:
        num_param=(depth+2)*4
        circuit=lambda x :circuit_22_Haar(scheme,depth,x)

    if scheme in ['NCDQML8','CCDQML8','QCDQML8']:
        num_param=(depth+2)*12
        circuit=lambda x :circuit_44_Haar(scheme,depth,x)

    F_list=[]
    starttime=time.time()
    for i in range (it):
        Param=RandParam(num_param)
        F_x=[]
        for j in range(x_sample):
            F_temp=Fisher(Param,circuit)
            F_x.append(F_temp)
        F_list.append(np.array(F_x))
        if i%10==0:
            donetime=time.time()
            print('iter',i+1,'done in',round(donetime-starttime,1),'sec')
    return F_list
def maxrank(FisherMatrix_list):  
    maxrank=0
    for F in FisherMatrix_list:
        rank=np.linalg.matrix_rank(F)
        if rank>maxrank:
            maxrank=rank
    return maxrank

def Fisher_Result(it,x_sample,depth_list,scheme):
    Rank_list=[]
    for depth in depth_list:
        starttime=time.time()
        F = Fisherinformation(scheme,depth,it,x_sample)
        F_xavg=np.mean(F,axis=1)
        r_m=maxrank(F_xavg)
        Rank_list.append(r_m)
        donetime=time.time()
        print('depth',depth,'done in',round(donetime-starttime,1),'sec')
    return Rank_list