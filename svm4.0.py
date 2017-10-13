import svm_functions as svm 
import numpy as np
import time
import cvxopt
start_time=time.time();
X,Y=svm.read_training_data();
X=svm.makeitv(X,Y);
Y=svm.multiclassY(Y)
K=svm.similarity_matrix(X);
print("time taken till this point",time.time()-start_time)
predictor1={};
predictor2={};
predictor3={};
predictor4={};
predictor5={};
predictor6={};
predictor7={};

for i in range(Y.shape[1]):
	temp=Y.T[i].T;
	##these lines are temporary and is reelvant to only first label and not others for now since store.txt contains lagrangean for only label 1 in one vs all classification
	
	# fr=open("store.txt","r")#
	# lagrange_multipliers=fr.read().split();
	# fr.close();
	# for i in range(len(lagrange_multipliers)):
	# 	lagrange_multipliers[i]=float(lagrange_multipliers[i])
	# lagrange_multipliers=np.array(lagrange_multipliers)
	
	#uptil these this willbe used right now for fast calculation
	
	lagrange_multipliers=svm.compute_multipliers(K,X,temp)##uncomment thiese lines when the temporary lines are removed to calculate the data
	
	# print(lagrange_multipliers)
	# print(lagrange_multipliers.size,"sdf")
	support_multipliers, support_vectors, support_vector_labels=svm.get_parameter_values(X,Y,lagrange_multipliers)
	# print(support_multipliers)
	# print(support_vectors)
	# print(support_vector_labels)
	print("time taken after ",i+1,"th lagrangian is ",time.time()-start_time)
	# print(support_vector_labels)
	max_index,max_value=svm.predict(X[7000],support_multipliers,support_vectors,support_vector_labels)
	predictor1[max_value]=max_index;
	max_index,max_value=svm.predict(X[6000],support_multipliers,support_vectors,support_vector_labels)
	predictor2[max_value]=max_index;
	max_index,max_value=svm.predict(X[5000],support_multipliers,support_vectors,support_vector_labels)
	predictor3[max_value]=max_index;
	max_index,max_value=svm.predict(X[3000],support_multipliers,support_vectors,support_vector_labels)
	predictor4[max_value]=max_index;
	max_index,max_value=svm.predict(X[2000],support_multipliers,support_vectors,support_vector_labels)
	predictor5[max_value]=max_index;
	max_index,max_value=svm.predict(X[1000],support_multipliers,support_vectors,support_vector_labels)
	predictor6[max_value]=max_index;	
	max_index,max_value=svm.predict(X[0],support_multipliers,support_vectors,support_vector_labels)
	predictor7[max_value]=max_index;


predictor1=sorted(predictor1.items())
predictor2=sorted(predictor2.items())
predictor3=sorted(predictor3.items())
predictor4=sorted(predictor4.items())
predictor5=sorted(predictor5.items())
predictor6=sorted(predictor6.items())
predictor7=sorted(predictor7.items())

print(predictor1,predictor2,predictor3,predictor4,predictor5,predictor6,predictor7)
print("total timetaken = ",time.time()-start_time)