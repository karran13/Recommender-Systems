'''
Created on Nov 4, 2017

@author: Binki
'''
import numpy as np
from math import sqrt
from numpy.core.umath import square

class matrixDec:
    
    def cosineSim(self,a,b):
        """Returns the cosine similarity score between two vectors a and b"""
        
        mod_a=0
        mod_b=0
        dot_ab=0
        for i in range(len(a)):
            dot_ab=a[i]*b[i] + dot_ab
            mod_a=a[i]*a[i] + mod_a
            mod_b=b[i]*b[i] + mod_b
        cos_ab=dot_ab/(sqrt(mod_a*mod_b))
        return cos_ab
    
    
    def svd(self,data_matrix):
        """Returns the SVD Decomposition of data_matrix in terms of U,sigma and V"""
        
        data_matrix_U=np.dot(data_matrix,data_matrix.T)
        data_matrix_V=np.dot(data_matrix.T,data_matrix)
        eig_val_U,eig_vect_U=np.linalg.eigh(data_matrix_U)
        eig_val_V,eig_vect_V=np.linalg.eigh(data_matrix_V)
        U=np.fliplr(eig_vect_U)
        V=np.fliplr(eig_vect_V)
        U=U/np.sqrt((np.sum(np.square(U),axis=0,keepdims=True)))
        V=V/np.sqrt((np.sum(np.square(V),axis=0,keepdims=True)))
        
        for i in range(U.shape[1]):
            if ( U[0][i] < 0 ):
                U[:,i]*=-1
        
        for i in range(V.shape[1]):
            if( V[0][i] < 0):
                V[:,i]*=-1
        
        eig_val_U=eig_val_U[::-1]
        sigma=np.sqrt(np.diag(eig_val_U))
        
        if(V.shape[0]>U.shape[0]):
            for i in range(V.shape[0]-U.shape[0]):
                sigma=np.c_[sigma,np.zeros(U.shape[0])] 
        if (V.shape[0]<U.shape[0]):
                for i in range(U.shape[0]-V.shape[0]):
                    sigma=np.delete(sigma,[sigma.shape[1]-1],axis=1)
        return U,sigma,V.T
    
    def reduceSvd(self,U,sigma,V):
        """Performs dimensionality reduction on given matrix decomposition till atleast 90% of energy is retained"""
        
        energy_matrix=np.array(np.diag(sigma))
        new_energy_matrix=energy_matrix
        while(1):
            new_energy_matrix=np.diag(np.delete(sigma,[sigma.shape[1]-1],axis=1))
            if(np.sum(np.square(new_energy_matrix))> 0.9*np.sum(np.square(energy_matrix))):
                sigma=np.delete(sigma,[sigma.shape[1]-1],axis=1)
                sigma=np.delete(sigma,[sigma.shape[0]-1],axis=0)
                U=np.delete(U,[U.shape[1]-1],axis=1)
                V=np.delete(V,[V.shape[0]-1],axis=0)
            else:
                break
            
        #print sigma.shape
        while(sigma.shape[1]!=sigma.shape[0]):
            if(sigma.shape[0]>sigma.shape[1]):
                sigma=np.delete(sigma,[sigma.shape[0]-1],axis=0)
                U=np.delete(U,[U.shape[1]-1],axis=1)
            if(sigma.shape[1]>sigma.shape[0]):
                sigma=np.delete(sigma,[sigma.shape[1]-1],axis=1)
                V=np.delete(V,[V.shape[0]-1],axis=0)
            #print sigma.shape
        return U,sigma,V
    
    def constEnergyReduction(self,U,sigma,V):
        """Performs dimensionality reduction on given matrix decomposition till atleast 90% of energy is retained"""
        
        energy_matrix=np.array(np.diag(sigma))
        new_energy_matrix=energy_matrix
        while(1):
            new_energy_matrix=np.diag(np.delete(sigma,[sigma.shape[1]-1],axis=1))
            if(np.sum(np.square(new_energy_matrix))== np.sum(np.square(energy_matrix))):
                sigma=np.delete(sigma,[sigma.shape[1]-1],axis=1)
                sigma=np.delete(sigma,[sigma.shape[0]-1],axis=0)
                U=np.delete(U,[U.shape[1]-1],axis=1)
                V=np.delete(V,[V.shape[0]-1],axis=0)
            else:
                break
            
        #print sigma.shape
        while(sigma.shape[1]!=sigma.shape[0]):
            if(sigma.shape[0]>sigma.shape[1]):
                sigma=np.delete(sigma,[sigma.shape[0]-1],axis=0)
                U=np.delete(U,[U.shape[1]-1],axis=1)
            if(sigma.shape[1]>sigma.shape[0]):
                sigma=np.delete(sigma,[sigma.shape[1]-1],axis=1)
                V=np.delete(V,[V.shape[0]-1],axis=0)
            #print sigma.shape
        return U,sigma,V

    
    
    
    def cur(self,data_matrix,n):
        """Performs C,U,R decomposition on data_matrix using n rows and columns"""
        
        frob_sum_matrix=np.sum(np.square(data_matrix))
        frob_sum_rows=np.sum(np.square(data_matrix),axis=1)
        frob_sum_cols=np.sum(np.square(data_matrix),axis=0)
    
        col_probs=[]
        row_probs=[]
        sel_rows=[]
        sel_cols=[]
    
        for i in range(len(frob_sum_rows)):
            row_probs.append(frob_sum_rows[i]/frob_sum_matrix)
    
        for i in range(len(frob_sum_cols)):
            col_probs.append(frob_sum_cols[i]/frob_sum_matrix)
        
        for i in range(n):
            sel_rows.append(np.random.choice(np.arange(len(row_probs)),p=row_probs))
    
        for i in range(n):
            sel_cols.append(np.random.choice(np.arange(len(col_probs)),p=col_probs))
        
        C=data_matrix[:,sel_cols]
        R=data_matrix[sel_rows]
        U=R[:,sel_cols]
        
        C=C/np.sqrt([col_probs[i] for i in sel_cols])
        C=C/np.sqrt(np.arange(1,len(sel_cols)+1))
        
        R=R/np.sqrt([row_probs[i] for i in sel_rows])[:,None]
        R=R/np.sqrt(np.arange(1,len(sel_rows)+1))[:,None]
        
        X,W,Y=self.svd(U)
        W=np.diag(W)
        W=np.reciprocal(W)
        W=np.diag(W)
        W=np.square(W)
        U_=np.dot(W,X.T)
        U=np.dot(Y.T,U_)
        return C,U,R
    
    def conceptSimRating(self,test_rating,i,j,data_matrix,concept_matrix,V):
        """Predicts the missing rating using matrix decomposition by finding the user most similar in concept who has rated the element"""
        
        user_concept=np.dot(test_rating,V.T)
        max_=0
        max_index=0
        for k in range(len(data_matrix)):
            if((i!=k) & (data_matrix[k][j]!=0.0)):
                cos_sim=self.cosineSim(user_concept,concept_matrix[k])
                if(cos_sim >= max_):
                    max_ = cos_sim
                    max_index = k
        
        test_rating[j]=data_matrix[max_index][j]
        return test_rating[j]
    
    def leaveOneOutAcc(self,data_matrix, V,n):
        """Prints RMS Error and Spearmans correlation on n sized random test data formulated by leave-one-out method"""
        
        concept_matrix=np.dot(data_matrix,V.T)
        sum_=0
        i,j=np.nonzero(data_matrix)
        ix=np.random.choice(len(i),np.floor(0.2*len(i)),replace=False)
        n=n%len(ix)
        for x in range(n):
            sum_+= square(self.conceptSimRating(data_matrix[i[ix[x]]],i[ix[x]], j[ix[x]],data_matrix,concept_matrix, V) - data_matrix[i[x]][j[x]])
            print self.conceptSimRating(data_matrix[i[ix[x]]],i[ix[x]], j[ix[x]],data_matrix,concept_matrix, V)
            print x
        sp_correl= 1 - 6*sum_/(n*(n*n-1))
        rmse= sqrt(sum_)/sqrt(n)
        print "RMSE:"
        print rmse
#        print sp_correl
        
    def precAtK(self,data_matrix,V,k):
        """Finds precision at k for given k, using given matrix decomposition"""
        
        concept_matrix=np.dot(data_matrix,V.T)
        relevant=0.0
        i,j=np.nonzero(data_matrix)
        ix=np.random.choice(len(i),np.floor(0.2*len(i)),replace=False)
        for x in range(k):
            rating=self.conceptSimRating(data_matrix[i[ix[x]]],i[ix[x]], j[ix[x]],data_matrix,concept_matrix, V)
            if ((data_matrix[i[x]][j[x]]>=3.5) & (rating>=3.5)):
                relevant+=1
        prec = relevant/(k)
        return prec


#A=np.array([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,2,0,4,4],[0,0,0,5,5],[0,1,0,2,2]])

