import svm_functions as svm 
import numpy as np
import time
start_time=time.time();
X,Y=svm.read_training_data();
X=svm.makeitv(X,Y);
Y=svm.multiclassY(Y)
K=svm.similarity_matrix(X);
temp=Y.T[0].T;
print("time taken till this point",time.time()-start_time)
lagrange_multipliers=svm.compute_multipliers(K,X,temp)#right now 17 steps;
print(time.time()-start_time)