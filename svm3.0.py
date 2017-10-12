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
def gaussian(sigma,x,y):
	print(x.shape,y.shape)
   	# return np.exp(-np.sqrt(abs(x-y) ** 2 / (2 * sigma ** 2)))

def _gram_matrix( X):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    # TODO(tulloch) - vectorize
    sigma=1e-5;
    for i, x_i in enumerate(X):
        for j, x_j in enumerate(X):
            K[i, j] = gaussian(sigma,x_i, x_j)
    return K
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

def _compute_multipliers( X, y):
    n_samples, n_features = X.shape

    K = _gram_matrix(X);
    # Solves
    # min 1/2 x^T P x + q^T x
    # s.t.
    #  Gx \coneleq h
    #  Ax = b

    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n_samples))

    # -a_i \leq 0
    # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
    G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    h_std = cvxopt.matrix(np.zeros(n_samples))

    # a_i \leq c
    G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
    h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

    G = cvxopt.matrix(np.vstack((G_std, G_slack)))
    h = cvxopt.matrix(np.vstack((h_std, h_slack)))

    A = cvxopt.matrix(y, (1, n_samples))
    b = cvxopt.matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return np.ravel(solution['x'])
def init_alltheta(nofeatures,labels):#no of features and labels
	return np.zeros((labels,nofeatures))

# def gaussian(sigma):
#     return lambda x, y: \
#         np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))



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
print(Y)
#arranging the data into X and Y matrix end
# print(X)

#initializing things start
continuous=0
#intializing things end

X=makeitv(X,Y,continuous)
# print(Y.shape)
Y=multiclassY(Y).T
# print(Y.shape)
print(Y)
m=X.shape[0]
n=X.shape[1]
labels=Y.shape[1]
w=init_alltheta(n,labels)
b=0
C=1;
print("here")
# print(Y.T[0].T)
print(X.shape)
lagranges_multiplier=_compute_multipliers(X,Y.T[0].T)
# # print(Y.shape)
# # P=np.outer(Y,Y)
# #
# # h=[C, 0]
# # h=np.asmatrix(h).T

# # G=[1,1]
# # G=np.asmatrix(G).T
# # print(G)
# n_samples=m;
# print(m)
# y=(Y.T)[0].T
# print(y.shape)
# K = np.zeros((n_samples, n_samples))
# P = cvxopt.matrix(np.outer(y,y) * K)
# print(P)
# q = cvxopt.matrix(np.ones(n_samples) * -1)
# A = cvxopt.matrix(y, (1,n_samples),'d')

# b = cvxopt.matrix(0.0)

# G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
# h = cvxopt.matrix(np.zeros(n_samples))

# #


# solution = cvxopt.solvers.qp(P, q, G, h, A, b)
print("time taken=",time.time()-start)



















# import numpy as np
# import cvxopt.solvers
# import logging


# MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


# class SVMTrainer(object):
#     def __init__(self, kernel, c):
#         self._kernel = kernel
#         self._c = c

#     def train(self, X, y):
#         """Given the training features X with labels y, returns a SVM
#         predictor representing the trained SVM.
#         """
#         lagrange_multipliers = self._compute_multipliers(X, y)
#         return self._construct_predictor(X, y, lagrange_multipliers)

#     def _gram_matrix(self, X):
#         n_samples, n_features = X.shape
#         K = np.zeros((n_samples, n_samples))
#         # TODO(tulloch) - vectorize
#         for i, x_i in enumerate(X):
#             for j, x_j in enumerate(X):
#                 K[i, j] = self._kernel(x_i, x_j)
#         return K

#     def _construct_predictor(self, X, y, lagrange_multipliers):
#         support_vector_indices = \
#             lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

#         support_multipliers = lagrange_multipliers[support_vector_indices]
#         support_vectors = X[support_vector_indices]
#         support_vector_labels = y[support_vector_indices]

#         # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
#         # bias = y_k - \sum z_i y_i  K(x_k, x_i)
#         # Thus we can just predict an example with bias of zero, and
#         # compute error.
#         bias = np.mean(
#             [y_k - SVMPredictor(
#                 kernel=self._kernel,
#                 bias=0.0,
#                 weights=support_multipliers,
#                 support_vectors=support_vectors,
#                 support_vector_labels=support_vector_labels).predict(x_k)
#              for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

#         return SVMPredictor(
#             kernel=self._kernel,
#             bias=bias,
#             weights=support_multipliers,
#             support_vectors=support_vectors,
#             support_vector_labels=support_vector_labels)

#     def _compute_multipliers(self, X, y):
#         n_samples, n_features = X.shape

#         K = self._gram_matrix(X)
#         # Solves
#         # min 1/2 x^T P x + q^T x
#         # s.t.
#         #  Gx \coneleq h
#         #  Ax = b

#         P = cvxopt.matrix(np.outer(y, y) * K)
#         q = cvxopt.matrix(-1 * np.ones(n_samples))

#         # -a_i \leq 0
#         # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
#         G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
#         h_std = cvxopt.matrix(np.zeros(n_samples))

#         # a_i \leq c
#         G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
#         h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

#         G = cvxopt.matrix(np.vstack((G_std, G_slack)))
#         h = cvxopt.matrix(np.vstack((h_std, h_slack)))

#         A = cvxopt.matrix(y, (1, n_samples))
#         b = cvxopt.matrix(0.0)

#         solution = cvxopt.solvers.qp(P, q, G, h, A, b)

#         # Lagrange multipliers
#         return np.ravel(solution['x'])


# class SVMPredictor(object):
#     def __init__(self,
#                  kernel,
#                  bias,
#                  weights,
#                  support_vectors,
#                  support_vector_labels):
#         self._kernel = kernel
#         self._bias = bias
#         self._weights = weights
#         self._support_vectors = support_vectors
#         self._support_vector_labels = support_vector_labels
#         assert len(support_vectors) == len(support_vector_labels)
#         assert len(weights) == len(support_vector_labels)
#         logging.info("Bias: %s", self._bias)
#         logging.info("Weights: %s", self._weights)
#         logging.info("Support vectors: %s", self._support_vectors)
#         logging.info("Support vector labels: %s", self._support_vector_labels)

#     def predict(self, x):
#         """
#         Computes the SVM prediction on the given features x.
#         """
#         result = self._bias
#         for z_i, x_i, y_i in zip(self._weights,
#                                  self._support_vectors,
#                                  self._support_vector_labels):
#             result += z_i * y_i * self._kernel(x_i, x)
#         return np.sign(result).item()