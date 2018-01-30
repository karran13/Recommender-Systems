# Recommender-Systems
Predicting movie ratings using collaborative filtering and latent concept extractors based on matrix decomposition methods like SVD and CUR on the movielens-100k dataset

The project explores techniques for predicting unknown user ratings based on a large dataset of users and movie ratings.

Movielens datasets were collected by the GroupLens Research Project at the University of Minnesota. 
 This data set consists of: 
 -100,000 ratings (1-5) from 943 users on 1682 movies.  
 -Each user has rated at least 20 movies.  
 -Simple demographic info for the users (age, gender, occupation, zip) 
 
 Methodology: 
 
We used 6 different recommender methods to predict user ratings across the data sets.  
 
Namely we set up Collaborative Filtering along with its baseline variant and Matrix decomposition techniques like Singular Value Decomposition and CUR. 
Moreover, we also implemented dimensionality reduction in these matrix decomposition techniques which preserved 90% of the energy so as to increase the computational efficiency while keeping the accuracy similar for these methods. 
