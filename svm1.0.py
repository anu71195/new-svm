import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import time
import sys
import math

def svmcost(X,Y,C,theta):
	h=np.matmul(X,theta);
	temp1=sum(np.multiply(Y,cost1(h,Y)))
	temp2=sum(np.multiply((1-Y),cost0(h,Y)))
	cost=C*(temp1+temp2)	+	(1/2)*(sum(	sum(np.multiply(theta,theta)).T	));
	return (cost);

def init_theta(X):
	return np.zeros((X.shape[0],1))
def cost_value_1(z):
	return(-(np.log(1/(1+np.exp(-z)))))
def cost_value_0(z):
	return (-(np.log(1-(1/(1+np.exp(-z))))))

def cost1(z,y):#y==1
	costmatrix=np.zeros((z.shape[0],1))
	y_1=np.multiply(y,z)
	for i in range(y_1.shape[0]):
		if z[i]<1 and z[i]!=0:
			costmatrix[i]=cost_value_1(z[i]);
	costmatrix=np.asmatrix(costmatrix)
	return costmatrix;


def cost0(z,y):
	costmatrix=np.zeros((z.shape[0],1))
	y_0=np.multiply(1-y,z)#y==0
	for i in range(y_0.shape[0]):
		if z[i]>-1:
			costmatrix[i]=cost_value_0(z[i]);
	return costmatrix;

def gradient(X,Y,theta):# x is features
	m=X.shape[0]
	h=cost_value_1(np.matmul(X,theta));
	theta_grad=(np.matmul((h-Y).T,X))/m;
	theta_grad=theta_grad.T
	return theta_grad;


def multiclassY(Y):
	columns=Y.shape[0]
	Y=np.asarray(Y);
	yvaluesdict={}
	yvalues=[]
	for i in Y:
		yvaluesdict[i[0]]=1
	for i in yvaluesdict:
		yvalues.append(i)
	yvalues.sort();
	rows=(len(yvaluesdict))
	Y=np.asmatrix(Y)
	Y_all=np.zeros((rows,columns))
	for index in range(rows):
		Y_all[index]=(Y==yvalues[index]).T
	Y_all=np.asmatrix(Y_all,dtype=int)
	return Y_all;


def init_alltheta(nofeatures,labels):#no of features and labels
	return np.zeros((labels,nofeatures))

def similarity_kernel(x,l,sigma):
	temp=l-x;
	return (np.exp((-sum((np.multiply(temp,temp)).T)/(2*sigma*sigma)))).T;

def svm_features(X,sigma,X_train):
	m=X.shape[0];
	n=X_train.shape[0]
	f=np.zeros((m,n))
	for i in range(m):
		f[i]=(similarity_kernel(X[i],X_train,sigma).T);
	return f;

def svm(X,Y,theta,alpha,C):
	m=X.shape[0]
	temp_cost=math.inf;
	i=0;
	while 1:
		theta=theta-(alpha/m)*(gradient(X, Y, theta))
		cost=svmcost(X,Y,C,theta)
		if cost>temp_cost:
			theta=temp_theta;
			print(i)
			break;
		temp_cost=cost;
		i=i+1;
		temp_theta=theta;
		print(cost,time.time()-start)
	return theta

def makeitv(X,Y,continuous):#make it velocity
	rows=X.shape[0]
	t=0.1;
	for i in range(rows):
		X[i]=X[i]*(t);
		t=t+0.01;
		if(continuous ==0 and i!=0 and Y[i]!=Y[i-1]):
			t=0.1;
	return X;
def test(file,all_theta,sigma,labels):
	fp=open(file,"r")#opening the data file
	X=[];
	Y=[];
	for i in fp:
		i=i.split(",")
		i.pop(0)
		i[3]=i[3].rstrip("\n")
		for j in range(len(i)):
			i[j-1]=int(i[j-1])
		Y.append(i[-1])
		i.pop(-1)
		X.append(i)
	X=np.asmatrix(X);	
	Y=np.asmatrix(Y);
	Y=Y.T;
	Y_orig=Y;
	#deleting this later just for trail
	temp=np.arange(30000,50000)
	X=X[temp];
	Y=Y[temp];
	print(X)
	continuous=0
	X=makeitv(X,Y,continuous);
	print(X)
	X=np.asmatrix(X)
	Y=np.asmatrix(Y)
	#deleting this later just for trial
	features=svm_features(X,sigma)
	Y=multiclassY(Y)
	h=np.matmul(features,all_theta.T)
	h_max=np.zeros((1,m))
	h_max[0]=np.amax(h,axis=1)
	h_max=h_max.T
	temp_h=h==h_max
	temp_h=np.asmatrix(temp_h,dtype=int).T
	for i in range(labels):
		temp_h[i]=temp_h[i]*(i+1);
	temp_h=temp_h.T
	temp_h=temp_h[temp_h>0].T
	print("prediction=",(sum(temp_h==Y_orig)/m)*100,"-----")

start=time.time()
fp=open("2.csv","r")#opening the data file
X=[];
Y=[];
for i in fp:
	print(i)
	i=i.split(",")
	i.pop(0)
	i[3]=i[3].rstrip("\n")
	for j in range(len(i)):
		i[j-1]=int(i[j-1])
	Y.append(i[-1])
	i.pop(-1)
	X.append(i)
X=np.asmatrix(X);	
Y=np.asmatrix(Y);
Y=Y.T;
argv=sys.argv;

#deleting this later just for trail
# temp=np.arange(30000,50000)
# X=X[temp];
# Y=Y[temp];
print(X)
print("hesadfsdre he")
continuous=0#0 means itis not continuous flow of data rather 
X=makeitv(X,Y,continuous);
print(X)
# X=np.asmatrix(X)
# Y=np.asmatrix(Y)
#deleting this later just for trial

#initializing all the data here
Y_orig=Y;#y_orig=yoriginal
theta=init_theta(X);
sigma=10;
C=0.00000001
alpha=10;
m=X.shape[0]
ran=np.arange(0.1,1,0.01)
testfile="1.csv"
#initializing the data done here

features=svm_features(X,sigma,X)
Y=multiclassY(Y)
labels=Y.shape[0]
all_theta=init_alltheta(features.shape[1],labels)
ytemp=np.zeros((1,m));
for i in range(labels):
	ytemp[0]=Y[i]
	all_theta[i]=svm(features,ytemp.T,theta,alpha,C).T
h=np.matmul(features,all_theta.T)
print(h.shape,"shape")
h_max=np.zeros((1,m))
h_max[0]=np.amax(h,axis=1)
h_max=h_max.T
temp_h=h==h_max

temp_h=np.asmatrix(temp_h,dtype=int).T
for i in range(labels):
	temp_h[i]=temp_h[i]*(i+1);
temp_h=temp_h.T
temp_h=temp_h[temp_h>0].T
print(temp_h.shape,Y_orig.shape)

print("prediction=",(sum(temp_h==Y_orig)/m)*100,"-----")
# test(testfile,all_theta,sigma,labels)

#crossvalidation part

train_features=X
fp=open("1.csv","r")
X=[];
Y=[];
for i in fp:
	i=i.split(",")
	i.pop(0)
	i[3]=i[3].rstrip("\n")
	for j in range(len(i)):
		i[j-1]=int(i[j-1])
	Y.append(i[-1])
	i.pop(-1)
	X.append(i)
X=np.asmatrix(X);
X=makeitv(X,Y,continuous);	
temp=np.arange(3000,6000)
X=X[temp]
Y=np.asmatrix(Y);
Y=Y.T;
Y=Y[temp]
m=X.shape[0]
print(m)
# print(Y)
features=svm_features(X,sigma,train_features)
print(features.shape)
print("features are")
print(features)

h=np.matmul(features,all_theta.T)
print("size of h",h.shape)
h_max=np.zeros((1,m))
print("size of h max",h_max.shape)
h_max[0]=np.amax(h,axis=1)
print("size of h max",h_max.shape)
print(h.shape)
h_max=h_max.T
print(h_max.shape)
print("here")
print(h.shape, h_max.shape)
temp_h=h==h_max
print(temp_h.shape)
temp_h=np.asmatrix(temp_h,dtype=int).T
print(temp_h.shape)
for i in range(labels):
	temp_h[i]=temp_h[i]*(i+1);
temp_h=temp_h.T
print(temp_h.shape)
print(temp_h)
temp_h=temp_h[temp_h>0].T
print(h)
print(temp_h.shape,Y.shape)
print("prediction=",(sum(temp_h==Y)/m)*100,"-----")