import time
import numpy as np
import cvxopt

def makeitv(X,Y,continuous):#make it velocity
	rows=X.shape[0]#rows are number of training examples
	t=0.1;
	for i in range(rows):
		X[i]=X[i]*(t);
		t=t+0.01;
		if(continuous ==0 and i!=0 and Y[i]!=Y[i-1]):
			t=0.1;
	return X;
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
	Y_all=(Y_all*2)-1
	return Y_all;

def init_alltheta(nofeatures,labels):#no of features and labels
	return np.zeros((labels,nofeatures))




start=time.time()#setting up the timer

#file processing for training data start
fr=open("2.csv","r")#opening the file
text=fr.read().split('\n')#reading the file and dividing into the array line wise
#file processing for training data end


#arranging the data into X and Y matrix start
X=[]#initiating empty training data
Y=[]#initiaintg empty labels for training data
for  line in text:
	if line=='':
		continue;
	line=line.split(',')
	Y.append(int(line.pop(-1)))
	line.pop(0)
	for subline in range(len(line)):
		line[subline]=int(line[subline])
	X.append(line)
X=np.asmatrix(X)
Y=np.asmatrix(Y).T
#arranging the data into X and Y matrix end
# print(X)

#initializing things start
continuous=0
#intializing things end

X=makeitv(X,Y,continuous)
Y=multiclassY(Y).T
print(Y)
m=X.shape[0]
n=X.shape[1]
labels=Y.shape[1]
w=init_alltheta(n,labels)
b=0
C=1;
# print(Y.shape)
# P=np.outer(Y,Y)
#
# h=[C, 0]
# h=np.asmatrix(h).T

# G=[1,1]
# G=np.asmatrix(G).T
# print(G)
n_samples=m;
print(m)
y=(Y.T)[0].T
print(y.shape)
K = np.zeros((n_samples, n_samples))
P = cvxopt.matrix(np.outer(y,y) * K)
print(P)
q = cvxopt.matrix(np.ones(n_samples) * -1)
A = cvxopt.matrix(y, (1,n_samples),'d')

b = cvxopt.matrix(0.0)

G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
h = cvxopt.matrix(np.zeros(n_samples))

#


solution = cvxopt.solvers.qp(P, q, G, h, A, b)
print("time taken=",time.time()-start)
