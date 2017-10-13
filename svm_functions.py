import time
import numpy as np 
import cvxopt#cvxopt is the convex optimization library
import svm_parameters as param#this library is built and all the parameters and other input will be put there and all other input functions  can be declaed in this file 

'''
to do list
create read me file

REad me file ;-
	add the label details
	assumption like the data is not continuous that is time is set to 0 whenever label is changed.
	assumption that the training data is taken at the difference of 0.1 seconds
'''


MIN_SUPPORT_VECTOR_MULTIPLIER=param.min_support_vector_multiplier


##function to read the file as given by the user
def read_training_data():
#name of the training file will be taken from the svm_parameters file and will be stored in the variable name training_file
	training_file=param.training_file;
#opening the file whose name is stored in the variable name training_file(name of the file is store in svm_parameters.py file)
	fr=open(training_file,"r");
#the training file is read and different lines are stored in an array
	text=fr.read().split();
# every line contains first element as the index or the number line then next three elements in the line are
#  x_direction, y_xirection and z_direciton acceleration respectively 
#  with the last element as label to the movement(see the read_me.txt)

# arranging the data into X and Y matrix where X will be the features of all the training data
# and Y will contains the labels of all those training examples
	X=[]#initiating empty training data 
	Y=[]#initiaintg empty labels for training data
	for  line in text:
# it will go index by index in the input taken from the training file as mentioned in svm_parameters.py
		if line=='':#skipping over empty inputs or null inputs
			continue;

# if input is not null then read that element (one at a time)in the input array(which is split line wise) the single element will be read as line
# then split that element into 5 different numbers separated by comma namely 
# first element is the index or the training example number starting from 0;
# second to fourth element are the acceleration in x,y and z direction 
# the fifth or the last element is the label given to the training data
# after splitting the data put the label into the Y array remove the index element  or the first element 
# and also remove the last or the label element and we are left with only features
# Now convert every element in the line(three elements which are element i.e. acceleration in all three directions
# x,y,z direction respectively) and then put it in the X array.
# Now convert X and Y from array into matrix. the number of rows in both X and Y will be same i.e. number of training examples
# the columns in Y will be 1 and in X will be 3 w.r.t to x, y, z direction accelerations.
# finally return X and Y i.e. features matrix X and label matrix Y;

		line=line.split(',')
		Y.append(int(line.pop(-1)))
		line.pop(0)
		for subline in range(len(line)):
			line[subline]=int(line[subline])
		X.append(line)
	X=np.asmatrix(X)
	Y=np.asmatrix(Y).T
	return X,Y;









def makeitv(X,Y):
#this function will turn the acceleration in the input to the velocity whenever new label is encountered
# the timer will restart that is time will be set to 0 and velocity will be count from that point more precisely
# whenever a new label is encountered we will freshly calculate the velocity from starting and then put it in an array
# and then return that array. here we have assumed that each training data of same label has a difference of 0.1 seconds
# we will be iterating over all the training examples and in each training examples 
	rows=X.shape[0]#rows are number of training examples
	t=0.1;#first data will be taken after 0.1 seconds after the timer start
	temp_y=[];
	for i in range(rows):
		temp_y.append(Y[i][0]);#making a dublicate array of Y
	temp_y.append(Y[rows-1]+1);#handling the label change when the label has to be chacked for the last training data
#t will always be constant because every data is been taken after 0.1seconds
	u=0;#initial velocity will be zero
	for i in range(rows):#iterating  over all the rows i.e. all the training examples
		X[i]=u+X[i]*(t);#v=u+at########u will always tell the current velocity
		u=X[i];#updating u into the current velocity;
		if(temp_y[i]!=temp_y[i+1]):#if labels changed then time is set to 0 i.e. velocity becomes zero
			u=0;
	return X;

def multiclassY(Y):
# the training data we have many different labels so basically it is not a binary classification but a multiclass classification
# so what we are doing is turning this classification as one vs all classification i.e. here the correct label corresponding
# to  the training sample is 1 and the otherwise instead  of 0  the other label is put -1 that means if we are talking
# about a some label say 1 which is some movement then corresponding to each sample the label will be having 1 if that 
# training example is the training example corresponding to the label 1 other wise it will show -1
	columns=Y.shape[0]#these will be number of rows in final output i.e. labels formulticlass classification
	Y=np.asarray(Y);#turning Y into array to itterate and get values properly
	yvaluesdict={}#initiating an emtpy dictionary it will store all the different labels or unique labels available in the input
	yvalues=[]#these will store the store labels from yvaluesdict
	for i in Y:#iterating over entire y
		yvaluesdict[i[0]]=1#storing different labels this is a dictionary
 # so duplicate entries are overwritten giving us only unique labels
	for i in yvaluesdict:#putting all these values in the array yvalues
		yvalues.append(i)
	yvalues.sort();#sorting all the labels in ascending order
	rows=(len(yvaluesdict))#these will be number of columns in the final output label matrix
	Y=np.asmatrix(Y)#turning label array back into matrix
	Y_all=np.zeros((rows,columns))#initiating output label matrix as zeros with rows and columns given as parameters
	for index in range(rows):
#iterating over each row here each row , here each row is an array of all the training examples and will store the value
# 1 for correct classification and -1 for wrong classification
		Y_all[index]=(Y==yvalues[index]).T#solving by comparison which ones are correct classification and which one are wrong
# if correct classification then 1 is written else 0 is written over there
	Y_all=np.asmatrix(Y_all,dtype=int)
#turning output matrix Y_all into matrix with vlaue type int from float64 dtype
	Y_all=(Y_all*2)-1#converting matrix of 1-0 to 1 and -1 . ... 1 corresponds to 1 and 0 correspondsto -1;
	return Y_all.T; #returning matrix of size number of examples X labels by transponsing



def gaussian(y,x,sigma):
# this is also a gaussian kernel similar to gaussian kernel howeer the only difference between gaussian and gaussian_kernel
# is that gaussian_kernel  computes similarity array or row at a time i.e. given a feature matrix X and a training 
# example x then it computes similarities between x and all the training examples in X and gives them as output
# in the row. However, gaussian simply computes similarity value beteen two training examples given x and y 
# it calculates similarity using gaussian kernel and return a single real positive value.
#gaussian kernel =exp( -( (x1-x2)^2)   /   (2*sigma*sigma) ) ) 
    return np.exp(    -( (y-x)*(y-x).T / (2 * sigma* sigma) )  )



def gaussian_kernel(x,l,sigma):
	temp=l-x;#x is a single trainnig exampleof size 1X3 and l is the entire feature matrix with samples*3 size
#gaussian kernel =exp( -( (x1-x2)^2)   /   (2*sigma*sigma) ) ) 
#this will calculate the similarity value for one training example x vs all l which is actually X which is training data
#then it will return an array of size equal to no of samples
#each index in that array will contain the similarity value for x and l[index];
	return (np.exp((-sum((np.multiply(temp,temp)).T)/(2*sigma*sigma)))).T;

def similarity_matrix(X):
#it gives similarity matrix i.e. similarity between every two pair of samples in the training data
	sigma=param.sigma;#sigma as mentioned by the user in the svm_parameters.py file
	m=X.shape[0];#no of training data=m because the size of similarity matrix will be mXm
	f=np.zeros((m,m))#initiating empty similarity matrix
	for i in range(m):
		f[i]=(gaussian_kernel(X[i],X,sigma).T);#updating similarity matrix index by index
	return f;#finally returning the similarity matrix;






def compute_multipliers(K, X, y):
    n_samples, n_features = X.shape
    #X.shape returns "rows columns" rows are stored in  n_samples and columsn are stored in n_features
    #here y is for particular label i.e. it is a binary classification for multiclass classification
    #what i did is made a Y one vs all label matrix and then pass y for one label at a time to get lagrangean 
    #for every particular type of label as present in the input;
    #K is the similarity matrix, X is the input matrix ;
    c=param.c
    # Solves
    # min 1/2 x^T P x + q^T x
    # s.t.
    #  Gx \coneleq h
    #  Ax = b
    y=np.asmatrix(y,dtype=float)
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n_samples))

    # -a_i \leq 0
    # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
    G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    h_std = cvxopt.matrix(np.zeros(n_samples))

    # a_i \leq c
    G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
    h_slack = cvxopt.matrix(np.ones(n_samples) *  c)

    G = cvxopt.matrix(np.vstack((G_std, G_slack)))
    h = cvxopt.matrix(np.vstack((h_std, h_slack)))

    A = cvxopt.matrix(y, (1, n_samples))
    b = cvxopt.matrix(0.0)

    # print(A.size)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return np.ravel(solution['x'])

def get_parameter_values(X,y,lagrange_multipliers):
    support_vector_indices = \
        lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
    support_multipliers = lagrange_multipliers[support_vector_indices]
    support_vectors = X[support_vector_indices]
    support_vector_labels = y[support_vector_indices]##support_vecto_labels is not a row or column matrix_check for validity
    return support_multipliers, support_vectors, support_vector_labels

def predict(x,support_multipliers,support_vectors,support_vector_labels,indexing):
	result=0;
	sigma=param.sigma;
	for z_i, x_i, y_i in zip(support_multipliers,support_vectors,support_vector_labels):
		# result += z_i * y_i[0] * gaussian(x_i, x,sigma)
		result+=(z_i * gaussian(x_i, x,sigma)* y_i[0] )
	result=np.array(result)
	result=result[0]
	return result[indexing] , indexing;###added another parameter indexing and corresponding to that this line
	max_value=-999999999;
	max_index=0;
	for i in range(len(result)):
		if max_value<result[i]:
			max_value=result[i];
			max_index=i;
	return max_index+1,max_value
