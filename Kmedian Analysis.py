# -*- coding: utf-8 -*-
"""
K median Analysis

@author: Siddhant Agarwal
"""

import numpy as np

def Kmedian(data, k, random_state = 0):
    
    '''
    Input - 
    data: The data in an array and not as a dataframe
    k: the number of clusters you require
    
    Output - depending on the breaking condition
    mu_new or mu: centroids of the clusters 
    assignment_new or assignment: the cluster assignment for each data point
    J_new or J_old: The loss value corresponding to a particular k (number of clusters)
    '''
    
    np.random.seed(random_state)
    # Randomly initializing cluster centroid from the data
    mu = data[np.random.randint(data.shape[0], size = k),:] 
    
    #Taking 1000 iterations as a maximum iterations if the algorithm doesn't converge
    for a in range(1000):
        
        distance_vector = np.zeros((k,data.shape[0]))
        for i in range(k):
            for j in range(data.shape[0]):
                #Calculating the manhattan distance of culster centroid with the data points
                distance_vector[i,j] = np.abs(mu[i] - data[j]).sum()

        #Assigning data points to the cluster based on its minimum distance from the centroid
        assignment = np.argmin(distance_vector, axis = 0)
    
        #Separating the index of the data points according to the cluster it is assigned
        index_cluster = [np.argwhere(i==assignment) for i in np.unique(assignment)]
    
        idx = []
        for i in range(k):
            idx1 = []
            for j in range(len(index_cluster[i])):
                idx1.append(index_cluster[i][j][0])
            idx.append(np.array(idx1))
    
        idx = np.array(idx)
    
        J_old = 0
        #Calculating cost function for the old assignment
        for i in range(k):
            for j in range(len(index_cluster[i])):
                J_old += np.abs(mu[i] - data[index_cluster[i][j][0]]).sum()
    
        J_old = J_old/data.shape[0]
    
        sorted_idx = [] #Sorting in ascending order based on the distance from the centroid
        
        #Calculating new centroid
        mu_new = np.zeros((k,data.shape[1]))
        for i in range(k):
            sorted_idx.append(idx[i][np.argsort(distance_vector[i, idx[i]])])
            mu_new[i, :] = np.median(data[sorted_idx[i]], axis = 0)
    
        distance_vector_new = np.zeros((k,data.shape[0]))
        for i in range(k):
            for j in range(data.shape[0]):
                #Calculating the manhattan distance of culster centroid with the data points
                distance_vector_new[i,j] = np.abs(mu_new[i] - data[j]).sum()
    
        #Assigning data points to the cluster based on its minimum distance from the centroid
        assignment_new = np.argmin(distance_vector_new, axis = 0)
    
        #Separating the index of the data points according to the cluster it is assigned
        index_cluster_new = [np.argwhere(i==assignment_new) for i in np.unique(assignment_new)]
    
        J_new = 0
        #Calculating cost function for the old assignment
        for i in range(k):
            try:
                for j in range(len(index_cluster_new[i])):
                    J_new += np.abs(mu_new[i]-data[index_cluster_new[i][j][0]]).sum()
            except:
                print('empty cluster')
            finally:
                J_new += 0
        J_new = J_new/data.shape[0]
        
        #Condition to break the initial for loop, convergence is achieved if the difference between the old and new cost function is greater than 0 and less than 1
        if (J_old - J_new) >=0 and (J_old - J_new) <1:
            return mu_new, assignment_new, J_new
            break
        #If old - new cost function is less than 0 then new cost is higher than the old cost thus return the older assignment 
        elif (J_old - J_new) <0:
            return mu, assignment, J_old
            break
        #Here convergence is not achieved thus assign new assignment as old assignment and continue till convergence is achieved
        else:            
            mu = mu_new
    
    return mu_new, assignment_new, J_new