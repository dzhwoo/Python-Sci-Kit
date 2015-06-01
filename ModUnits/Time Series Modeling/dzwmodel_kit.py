#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:      Over the course, would like to build out a data science kit. Where I can apply different methods to data sets
#
# Author:      dwoo57
#
# Created:     16/01/2015
# Copyright:   (c) dwoo57 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import helper
import csv
import numpy as np
import random
import math
import matplotlib.pylab as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

tweetsInputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Output_tweets_interval_rates_trending_topics_2015_0111_to_0125_V2.csv"
#tweetsOutputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Cluster_Output_tweets_interval_rates_trending_topics_2015_0111_to_0125_V2.csv"
tweetsclOutputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Cluster_Groups_Output_tweets_interval_rates_trending_topics_2015_0111_to_0125_V2.csv"
#tweetsRowHeaderOutputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Cluster_Groups_Output_tweets_rowheader.csv"
#tweetsRowIndexOutputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Cluster_Groups_Output_tweets_rowindex.csv"
#tweetsColumnHeaderOutputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Cluster_Groups_Output_tweets_columnheader.csv"
#tweetsColumnIndexOutputFilePath = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Scipts\\Analysis\\Cluster_Trends_0111_to_0125_2_week\\Cluster_Groups_Output_tweets_columnindex.csv"

def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)

# if sample size is 0 then no random sampling
# What is the flow here
# Also, would like it if debug to show step by step

"""
THe flow is as follows:
    1. Initial centriods - using random sample of n, where n is the # of clusters
    2.

"""
def k_means_clust(groups_dtw,data,num_clust,num_iter,sample_size,w=5):

    isDebugOn = True

    centroids=random.sample(data,num_clust)
    counter=0

    if sample_size != 0:
        data = random.sample(data,sample_size)



    #for i in range(len(data)):
        #plt.plot(data[i])

    #plt.show()

    groups_topic ={}

    for n in range(num_iter):
        counter+=1
        print counter
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]

        #recalculate centroids of clusters

        # here assigments is the set of clusters. or assignments = clusterss
        #1. For each cluster
        for key in assignments:
            clust_sum=0
            #2. key =clusters. Sum all data points within cluster. Is it the sum or should we use the normalized points after time warping? But time warping needs to be done relatively to something

            # this is in the case that this cluster does not have any topics.
            if len(assignments[key]) < 1:
                if False:
                    print "Centriod had not topics associated with it"
                    print centroids[key]
                    plt.plot(centroids[key])
                    plt.show() #show orginal clusters
                continue
            else:
                for k in assignments[key]:
                    clust_sum=clust_sum+data[k]

                    # for last iteration print groups out
                    if n == num_iter -1 :
                        if k not in groups_topic:
                            groups_topic[k] = key

            #3. Recalc the current centriod based on the average of topics. For each interval, sum across topics and divide by number of topics.
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]

        # then need to sort dictionary by topic and print groups out
        for key in sorted(groups_topic.iterkeys(),reverse = False):
            #groups_dtw[key] = groups_topic[key]
            # index is always rows then columns
            # we want topic then cluster
            groups_dtw[key,0] = key
            groups_dtw[key,1] = groups_topic[key]
            #groups_dtw[key] = groups_topic[key]

    # Visualize final plots. This is visual inspection of clusters.
    #isDebugOn = True
    if isDebugOn == True:

        index = 0
        for i in centroids:
            plt.plot(i)
            plt.show() #show orginal clusters
            if len(assignments[index]) > 0:
                for topics in assignments[index]:
                    print assignments[index]
                    plt.plot(data[topics])

            plt.show()
            index +=1


    return centroids,groups_dtw

#create function to reorder matrix based on column values
def Take2dArrayOrderByColumnHeader(inputarray,columnlabels,rowlabels):

    #outarray = np.zeros((len(rowlabels), len(columnlabels)), dtype = 'f4')
    #outarray = np.zeros((len(rowlabels), 121), dtype = 'f4')
    outarray = np.zeros((len(rowlabels), len(columnlabels) + 1 ), dtype = 'f4')

    index = 0
    for label in columnlabels:
        print index,label
        if label != 'foo':
            outarray[:,int(float(label))] = inputarray[:,index]
        index +=1

    return outarray

# This calculates the silhouttescore or whether clusters overlap using dynamic time wrapping euclidean distance
def CalcSilhoutteScoreUsingDTW(X, labels):

    #1. for each row calculate the pairwise distance between another row
    distances = pairwise_distances(X, metric=metric, **kwds)
    n = labels.shape[0]
    A = np.array([_intra_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    B = np.array([_nearest_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    sil_samples = (B - A) / np.maximum(A, B)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)

# Paired distances
def paired_euclidean_distances(X, Y):
    """
    Computes the paired euclidean distances between X and Y

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray (n_samples, )
    """
    X, Y = check_paired_arrays(X, Y)

    return np.sqrt(((X - Y) ** 2).sum(axis=-1))

def KMeansClustBasedOnDynamicTimeWrapping(tweetsInputFilePath,num_clust,sample_size,tweetsclOutputFilePath,tweetsclCentriodsOutputFilePath,tweet_rate_col_idx):
    #data = helper.ImportFileConvertToNumpyArray(tweetsInputFilePath,0,',','a10,f4,f4,f4,f4')
    data = helper.ImportCSVFileConvertToNumpyArray(tweetsInputFilePath)
    #data = data[:,[0,1,4]]
    data = data[:,[0,1,tweet_rate_col_idx]]
    #reader = csv.reader( open(tweetsInputFilePath) )

    # this gets unique rows?
    rows, row_pos = np.unique(data[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)
    rows, row_pos = np.unique(data[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(rows), len(cols)), dtype = 'f4')
    #pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    pivot_table[row_pos, col_pos] = data[:, 2]
    data_pivoted = pivot_table

    data_pivoted_colsorted = Take2dArrayOrderByColumnHeader(data_pivoted,cols,rows)

    #we wanted to print the topics and clusters
    #groups_dtw = np.zeros(len(rows), dtype = 'f4')
    groups_dtw = np.zeros((len(rows),2), dtype = 'f4')

    num_iter = 10
    centroids,groups_dtw =k_means_clust(groups_dtw,data_pivoted_colsorted,num_clust,num_iter,sample_size,4)

    groups_dtw_mini = groups_dtw[:,0]


    #score = silhouette_score(data_pivoted_colsorted,groups_dtw)
    score = silhouette_score(data_pivoted_colsorted,groups_dtw_mini)
    print score

    #groups_dtw is sorted
    #TODO, print topic, then cluster as well to txt file
    np.savetxt(tweetsclOutputFilePath, groups_dtw)
    np.savetxt(tweetsclCentriodsOutputFilePath, centroids)

    for i in centroids:
        plt.plot(i)

    plt.show()

def LoadTweetsIntervalRatesIntoPivotTable(tweetsInputFilePath,tweet_rate_col_idx):

    isDebug = True

    #data = helper.ImportFileConvertToNumpyArray(tweetsInputFilePath,0,',','a10,f4,f4,f4,f4')
    data = helper.ImportCSVFileConvertToNumpyArray(tweetsInputFilePath)
    #data = data[:,[0,1,4]]
    data = data[:,[0,1,tweet_rate_col_idx]]
    #reader = csv.reader( open(tweetsInputFilePath) )

    # this gets unique rows?
    rows, row_pos = np.unique(data[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)
    rows, row_pos = np.unique(data[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)



    if isDebug == True:

        working_folder_output = "C:\\Users\\dwoo57\\Google Drive\\Career\\Projects\\Trending Topics\\Modeling\\Cluster_Trends_0111_to_0125_2_week\\3 clusters 10 Iterations\\Input Files\\"
        topic_file_path = working_folder_output + "Output_TEST_trending_topics.csv"
        np.savetxt(topic_file_path, rows,fmt="%s")

    pivot_table = np.zeros((len(rows), len(cols)), dtype = 'f4')
    #pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    pivot_table[row_pos, col_pos] = data[:, 2]
    data_pivoted = pivot_table

    data_pivoted_colsorted = Take2dArrayOrderByColumnHeader(data_pivoted,cols,rows)

    groups_dtw = np.zeros(len(rows), dtype = 'f4')

    return data_pivoted_colsorted, groups_dtw

#This is different from above. Maybe rename above to fit. This takes the clusters from above and then assigns them. Also, added option to use weight mean distance
def Predict_k_means_clust(centroids,tweets,groups_dtw,isUseExpWeightedMean,w=5):
    #centroids=random.sample(data,num_clust)
    #counter=0

    groups_topic ={}
    assignments={}
    topic_cluster_dist={-99:{-99:1.000}}
    #topic_tweet_dict = {'dict1':{'foo':1}}

    #1.Import centriods
    #2.Import tweet rate

    if isUseExpWeightedMean == False:

        #3.Then assign each topics to the nearest centriod
        for ind,i in enumerate(tweets):
                min_dist=float('inf')
                closest_clust=None
                for c_ind,j in enumerate(centroids):
                    if LB_Keogh(i,j,5)<min_dist:
                        cur_dist=DTWDistance(i,j,w)
                        if cur_dist<min_dist:
                            min_dist=cur_dist
                            closest_clust=c_ind
                if closest_clust in assignments:
                    assignments[closest_clust].append(ind)
                else:
                    assignments[closest_clust]=[]

        for key in assignments:
            for k in assignments[key]:
                if k not in groups_topic:
                            groups_topic[k] = key

        #4.Return assignments, for each topic what group do they belong to
        for key in sorted(groups_topic.iterkeys(),reverse = False):
                groups_dtw[key] = groups_topic[key]


        #5. Then plot centroids and their topics
        index = 0
        for i in centroids:

            plt.plot(i)
            plt.show() #show orginal clusters
            if assignments.has_key(index):
                for topics in assignments[index]:
                    print assignments[index]
                    plt.plot(tweets[topics])

                plt.show()
                print index, len(assignments[index])
            index +=1

        return groups_dtw

    else:
        #a. Calulate distance between each clusters and calc exponential. In this case, lower the better. Goal is to minimize the distance
        #b. Then sum trending and non trending and take the ratio of trending/nontrending
        #c. Results: if ratio <1 then trending, >1 then nontrending.
        #d. Store distance between each cluster and each topics. So should have table like topic, cluster 1 dist, cluster 2 dist, cluster 3 dist

        #a i). Per topic, iterate through each cluster and calc the distance
        for topic,topic_interval_rates in enumerate(tweets):

                min_dist=float('inf')
                closest_clust=None

                if topic != 15:
                    continue



                for cluster,cluster_interval_rates in enumerate(centroids):

                        cur_dist=DTWDistance(topic_interval_rates,cluster_interval_rates,w)

                        #a ii) if cluster is new then add
                        if topic in topic_cluster_dist:
                            #topic_cluster_dist[topic].append(cur_dist)
                            topic_cluster_dist[topic][cluster] = cur_dist
                        else:
                            #topic_cluster_dist[topic] = [cluster,cur_dist]
                            topic_cluster_dist[topic] = {cluster:cur_dist}

                #break;


        #bi) next calculate the sum of exponential distance for each trending clusters
        running_sum_exp_dist_trending = {}

        for topic in topic_cluster_dist:

            if topic == -99:
                continue
            #TODO replace with the number of trending topics
            for cluster in range(0,3):
                temp_exp_dist_trending = math.exp(topic_cluster_dist[topic][cluster])

                if topic in running_sum_exp_dist_trending:
                    running_sum_exp_dist_trending[topic] = temp_exp_dist_trending + running_sum_exp_dist_trending[topic]
                else:
                    running_sum_exp_dist_trending[topic] = temp_exp_dist_trending

        #bii) next calculate the sum of exponential distance for each NON-trending clusters
        running_sum_exp_dist_nontrending = {}

        for topic in topic_cluster_dist:
            if topic == -99:
                continue
            #TODO replace with the number of trending topics
            for cluster in range(4,7):
                temp_exp_dist_trending = math.exp(topic_cluster_dist[topic][cluster])

                if topic in running_sum_exp_dist_nontrending:
                    running_sum_exp_dist_nontrending[topic] = temp_exp_dist_trending + running_sum_exp_dist_nontrending[topic]
                else:
                    running_sum_exp_dist_nontrending[topic] = temp_exp_dist_trending

        #biii) now for each topic calculate the ratio
        weighted_mean_distance_per_topic = {}

        for topic in running_sum_exp_dist_trending:
            if topic == -99:
                continue
            #TODO replace with the number of trending topics
            weighted_mean_distance_per_topic[topic] = running_sum_exp_dist_trending[topic]/running_sum_exp_dist_nontrending[topic]
            print topic,running_sum_exp_dist_trending[topic]/running_sum_exp_dist_nontrending[topic]


        return

def CalcLinearInterpolation(firstXvalue,firstYvalue,secondXvalue,secondYvalue, currentXvalue):
    # Y1 - Y2/ (X1 - X2) * XC

    #return (following_rate - prev_rate)/(prev_interval - following_rate_interval ) * (prev_interval - interval)
    return (secondYvalue - firstYvalue)/(secondXvalue - firstXvalue ) * (currentXvalue - firstXvalue) + firstYvalue

def PredictAssignClosestClusterBasedOnDynamicTW(tweetsclCentriodsOutputFilePath_trend,tweetsclCentriodsOutputFilePath_nontrend, loc_output_TRENDING_tweetrate_TEST,tweet_rate_col_idx):

    #1a Load trending clusters
    #1b Load nontrending clusters
    pivot_table_trending = np.loadtxt(tweetsclCentriodsOutputFilePath_trend)
    pivot_table_nontrending = np.loadtxt(tweetsclCentriodsOutputFilePath_nontrend)

    pivot_table_clusters = np.concatenate((pivot_table_trending, pivot_table_nontrending))

    #2.load data and then Assign topics to closest cluster
    tweets, groups_dtw = LoadTweetsIntervalRatesIntoPivotTable(loc_output_TRENDING_tweetrate_TEST,tweet_rate_col_idx)

    #3.Then return topics and which groups they belong to
    isUseExpWeightedMean = False
    groups_dtw = Predict_k_means_clust(pivot_table_clusters,tweets,groups_dtw,isUseExpWeightedMean)

    return

#if __name__ == '__main__':
#    main()
