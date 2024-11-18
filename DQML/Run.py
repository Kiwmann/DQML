from .Machine import DQML 
import pennylane as qml
from pennylane import numpy as np
import pickle

def run_DQML(scheme,depth,X,iter,iter_train,iter_start=0):

    for i in range(iter_start,iter_start+iter):
        C,A,W,B=DQML(scheme,depth,iter_train,X)
        Result=[C,A,W,B]
        with open(scheme+'_'+'depth'+str(depth)+'_'+str(i+1), 'wb') as f:
            pickle.dump(Result, f)