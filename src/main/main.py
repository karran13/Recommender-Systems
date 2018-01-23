import pandas as pd
import numpy as np
from main.collab import collabFilt
from main.svd import matrixDec    

"""Main script which formats the data and runs the different recommender system methods, 
giving statistics for their performance"""


MAIN_DATA = "C:\\Users\\Binki\\Documents\\Work (question)\\BITS\\Academics\\4th year\\IR\\Assignment 3\\ml-100k\\u.data"
dataset_file = open(MAIN_DATA)
dataset_train = pd.read_csv(dataset_file, delim_whitespace=True)
# dataset_test=pd.read_csv(dataset_file,delim_whitespace=True)

collabfilter_1 = collabFilt()
matrixDec_1 = matrixDec()

data_matrix_train = collabfilter_1.data_format(dataset_train)

#A=np.array([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,2,0,4,4],[0,0,0,5,5],[0,1,0,2,2]])
# 
#U,sigma,V = matrixDec_1.svd(A)
# 
#sigma= np.nan_to_num(sigma)
# 
#U,sigma,V = matrixDec_1.constEnergyReduction(U, sigma, V)
# 
#print U,sigma,V


# collabfilter_1.precAtK(data_matrix_train, 100)
# collabfilter_1.precAtKbase(data_matrix_train, 100)
  
  
print "First 100 results and rmse and spearman for Collab Filtering"
collabfilter_1.leaveOneOutAcc(data_matrix_train, 100)
print "First 100 results and rmse and spearman for Collab Filtering with baseline"
collabfilter_1.leaveOneOutAcc_base(data_matrix_train, 100)
  
 
 
 
 
data_matrix_train = np.loadtxt('test1.txt', delimiter=',')
   
   
print "First 100 results and rmse and spearman for SVD"
U, sigma, V = matrixDec_1.svd(data_matrix_train)
print sigma.shape
sigma = np.nan_to_num(sigma)
U,sigma,V = matrixDec_1.constEnergyReduction(U, sigma, V)
print sigma.shape
matrixDec_1.leaveOneOutAcc(data_matrix_train, V, 100)
   
print "First 100 results and rmse and spearman for reduced SVD"
U, sigma, V = matrixDec_1.reduceSvd(U, sigma, V)
print "new SVD shape:"
print sigma.shape
   
matrixDec_1.leaveOneOutAcc(data_matrix_train, V, 100)
   
   
print "First 100 results and rmse and spearman for CUR"
U, sigma, V = matrixDec_1.cur(data_matrix_train, 100)
   
matrixDec_1.leaveOneOutAcc(data_matrix_train, V, 100)
   
   
print "First 100 results and rmse and spearman for reduced CUR"
U, sigma, V = matrixDec_1.reduceSvd(U, sigma, V)
#matrixDec_1.precAtK(data_matrix_train, V, 100)
   
matrixDec_1.leaveOneOutAcc(data_matrix_train, V, 100)

# print np.diag(sigma)
# print sigma.shape[1]

# print matrixDec_1.precAtK(data_matrix_train, V, 100)
