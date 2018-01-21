'''
Created on Nov 8, 2017

@author: Binki
'''
import pandas as pd
import numpy as np
import heapq
from math import sqrt
from numpy.core.umath import square

class collabFilt:

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
    
    
    
    def findRating(self,i,j,n,data_matrix):
        """Finds rating of given element based on data_matrix using user-user collaborative filtering"""
        
        rating=0
        sim_list=[]
        rating_wt_list=[]
        sim_sum=0
        rate_sum=0
        for k in range(len(data_matrix)):
            if((i!=k) & (data_matrix[k][j]!=0.0)):
                cos_sim=self.cosineSim(data_matrix[i],data_matrix[k])
                rat_wt=cos_sim*data_matrix[k][j]
                sim_list.append(cos_sim)
                rating_wt_list.append(rat_wt)
        max_index=heapq.nlargest(n, range(len(sim_list)), sim_list.__getitem__)
        for m in range(len(max_index)):
            rate_sum = rate_sum + rating_wt_list[max_index[m]]
            sim_sum = sim_sum + sim_list[max_index[m]]
        if(sim_sum==0):
            rating=rate_sum/n
        else:
            rating=rate_sum/sim_sum
        return rating
    
    
    
    def findRatingBase(self,i,j,n,data_matrix,matrix_mean,row_mean,col_mean):
        """Finds rating of given element based on data_matrix using user-user collaborative filtering with baseline approach"""
        
        rating=0
        sim_list=[]
        rating_wt_list=[]
        sim_sum=0
        rate_sum=0
        base_rating= matrix_mean + (row_mean[i]-matrix_mean) + (col_mean[j]-matrix_mean)
        for k in range(len(data_matrix)):
            if((i!=k) & (data_matrix[k][j]!=0.0)):
                cos_sim=self.cosineSim(data_matrix[i],data_matrix[k])
                rat_wt=cos_sim*(data_matrix[k][j] - (matrix_mean + (row_mean[k]-matrix_mean) + (col_mean[j]-matrix_mean)))
                sim_list.append(cos_sim)
                rating_wt_list.append(rat_wt)
        max_index=heapq.nlargest(n, range(len(sim_list)), sim_list.__getitem__)
        for m in range(len(max_index)):
            rate_sum = rate_sum + rating_wt_list[max_index[m]]
            sim_sum = sim_sum + sim_list[max_index[m]]
        if(sim_sum==0):
            rating=rate_sum/n
        else:
            rating = base_rating + rate_sum/sim_sum
        return rating
    
    
    def data_format(self,dataset):
        """Formats the given dataset into user-movie matrix format"""
        
        dataset=dataset[['UserId','MovieId','Rating']]
        
        data_matrix = [[0 for i in range(1682)] for j in range(943)]
        for i in range(943):
            dataset_temp=dataset[dataset['UserId']==i+1]
            for index,row in dataset_temp.iterrows():
                data_matrix[i][row['MovieId']-1]=row['Rating']
    
        dataframe_matrix=pd.DataFrame(data_matrix)
        dataframe_matrix['mean']= dataframe_matrix.mean(axis=1)
        
    #    user_mean_matrix=dataframe_matrix['mean']
    #    print user_mean_matrix.describe()
            
    #    for i in range(943):
    #        for j in range(1682):
    #            if(data_matrix[i][j]==0.0):
    #                print i
    #                print j
    #                data_matrix[i][j]=user_mean_matrix[i]
        np.savetxt('test1.txt', data_matrix,delimiter=',',newline='\n')
        return data_matrix
    
    def leaveOneOutAcc_base(self,data_matrix,n):
        """Prints RMS Error and Spearmans correlation on n sized random test data formulated by leave-one-out method"""
        
        sum_=0
        i,j=np.nonzero(data_matrix)
        ix=np.random.choice(len(i),np.floor(0.2*len(i)),replace=False)
        n=n%len(ix)
        
        matrix_mean=np.mean(data_matrix)
        row_mean=np.mean(data_matrix,axis=1)
        col_mean=np.mean(data_matrix,axis=0)

        for x in range(n):
            sum_+= square(self.findRatingBase(i[ix[x]], j[ix[x]], 5, data_matrix,matrix_mean,row_mean,col_mean) - data_matrix[i[x]][j[x]])
            print self.findRatingBase(i[ix[x]], j[ix[x]], 5, data_matrix,matrix_mean,row_mean,col_mean)
            print x
        sp_correl= 1 - 6*sum_/(n*(n*n-1))
        rmse= sqrt(sum_)/sqrt(n)
        print "RMSE:"
        print rmse
#        print sp_correl
    
    def leaveOneOutAcc(self,data_matrix,n):
        """Prints RMS Error and Spearmans correlation on n sized random test data formulated by leave-one-out method"""
        
        sum_=0
        i,j=np.nonzero(data_matrix)
        ix=np.random.choice(len(i),np.floor(0.2*len(i)),replace=False)
        n=n%len(ix)
        for x in range(n):
            sum_+= square(self.findRating(i[ix[x]], j[ix[x]], 5, data_matrix) - data_matrix[i[x]][j[x]])
            print self.findRating(i[ix[x]], j[ix[x]], 5, data_matrix)
            print x
        sp_correl= 1 - 6*sum_/(n*(n*n-1))
        rmse=sqrt(sum_)/sqrt(n)
        print "RMSE:"
        print rmse
 #       print sp_correl
        
        
    def precAtK(self,data_matrix,k):
        """Finds precision at k for given k, using collaborative filtering"""
        
        relevant=0.0
        i,j=np.nonzero(data_matrix)
        ix=np.random.choice(len(i),np.floor(0.2*len(i)),replace=False)
        for x in range(k):
            rating=self.findRating(i[ix[x]], j[ix[x]], 5, data_matrix) 
            if((data_matrix[i[x]][j[x]]>=3.5) & (rating>=3.5)):
                relevant+=1 
        prec = relevant/k
        print prec
    
    
    def precAtKbase(self,data_matrix,k):
        """Finds precision at k for given k, using collaborative filtering with baseline approach"""
        
        relevant=0.0
        matrix_mean=np.mean(data_matrix)
        row_mean=np.mean(data_matrix,axis=1)
        col_mean=np.mean(data_matrix,axis=0)
        i,j=np.nonzero(data_matrix)
        ix=np.random.choice(len(i),np.floor(0.2*len(i)),replace=False)
        for x in range(k):
            rating=self.findRatingBase(i[ix[x]], j[ix[x]], 5, data_matrix,matrix_mean,row_mean,col_mean) 
            if((data_matrix[i[x]][j[x]]>=3.5) & (rating>=3.5)):
                relevant+=1 
        prec = relevant/k
        print prec
