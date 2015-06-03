#-------------------------------------------------------------------------------
# Name:        sci_kit_features
# Purpose:      Thinking of the data science pipeline ( clean, feature, model, score)
#               This will probably be all things used to do feature engineering
#
# Author:      dwoo57
#
# Created:     14/01/2015
# Copyright:   (c) dwoo57 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import re
import numpy
from datetime import datetime
import csv
import random
from matplotlib import pyplot as PLT


def Multiplotter(num_row,num_col,num_plots,data):

    fig = PLT.figure()

    #211 (which could also be written in 3-tuple form as (2,1,1) means two rows of plot windows; one column; the third digit specifies the positioning relative to the other subplot windows--
    # rows, column, positioning relative to others
    # these means prob can only do 9 plots at a time

    base_subplot_num = num_row * 100 + num_col * 10

    for i in range(len(data)):

        ax = fig.add_subplot(base_subplot_num + i + 1)
        ax.plot(data[i])

    PLT.show()

def Multiplotter_MultipleDataset(cluster_centriods, cluster_assignments,assignments_data,num_row,num_col,num_plots):
     # Here could potentially create a function that plots the plots after each iteration
    """
    Steps:
        1. Get centriods ( First dataset)
        2. For each centriod, get assignments or topics. ( second dataset)
        3. For each topic, than plot this. ( third dataset)
        4.
    """
    if True:
        fig = PLT.figure()
        base_subplot_num = num_row * 100 + num_col * 10

        index = 0
        for i in cluster_centriods:

            ax = fig.add_subplot(base_subplot_num + index + 1)
            ax.plot(i)

            #plt.plot(i)
            #plt.show() #show orginal clusters
            if len(cluster_assignments[index]) > 0:
                for topics in cluster_assignments[index]:
                    #print cluster_assignments[index]
                    #plt.plot(assignments_data[topics])
                    ax.plot(assignments_data[topics])

            #plt.show()
            index +=1

        PLT.show()

def Multiplotter_MultipleDataset_WTable(cluster_centriods, cluster_assignments,assignments_data,num_row,num_col,num_plots):
     # Here could potentially create a function that plots the plots after each iteration
     # This was helpful when plotting, http://stackoverflow.com/questions/28505315/two-tables-in-matplotlib
    """
    Steps:
        1. Get centriods ( First dataset)
        2. For each centriod, get assignments or topics. ( second dataset)
        3. For each topic, than plot this. ( third dataset)
        4.
    """
    if True:
        fig = PLT.figure()
        base_subplot_num = num_row * 100 + num_col * 10

        cell_text=[]
        topic_list=[]
        index = 0
        for i in cluster_centriods:

            ax = fig.add_subplot(base_subplot_num + index + 1)
            ax.plot(i)

            #plt.plot(i)
            #plt.show() #show orginal clusters
            if len(cluster_assignments[index]) > 0:
                for topics in cluster_assignments[index]:
                    #print cluster_assignments[index]
                    #plt.plot(assignments_data[topics])
                    ax.plot(assignments_data[topics])

                    # this stores the table details. Do topics than cluster
                    cell_text.append([topics,index])

                    topic_list.append([topics])



            #plt.show()
            index +=1

        # Add a table at the bottom of the axes

        # this pushes the graphs to the left more
        PLT.subplots_adjust(right=0.7)

        # below plots the table
        #sets the posittion of the table
        left, width = 0.1, 0.6
        bottom, height = 0.1, 0.8
        left_table = left+width+0.1
        table_width = 0.15
        table_height = width/2.

        rect_table1 = [left_table, table_height+bottom , table_width, table_height]

        axTable1 = PLT.axes(rect_table1, frameon =False)
        axTable1.axes.get_yaxis().set_visible(False)

        axTable1.table(cellText=cell_text,
                              #rowLabels=rows,
                              #rowColours=colors,
                              colLabels=["topics","clusters"],
                              loc='upper center')

        print "Number of topics after each clustering"
        print len(topic_list)
        print len(assignments_data) # this is ok

        PLT.show()


def Multiplotter_KmeansStandardSciKit(cluster_centriods, cluster_assignments,assignments_data,num_row,num_col,num_plots):
     # Here could potentially create a function that plots the plots after each iteration
     # This was helpful when plotting, http://stackoverflow.com/questions/28505315/two-tables-in-matplotlib
    """
    Steps:
        1. Get centriods ( First dataset)
        2. For each centriod, get assignments or topics. ( second dataset)
        3. For each topic, than plot this. ( third dataset)
        4.
    """
    if True:
        fig = PLT.figure()
        base_subplot_num = num_row * 100 + num_col * 10

        cell_text=[]
        topic_list=[]
        cluster_index = 0
        for i in cluster_centriods:


            ax = fig.add_subplot(base_subplot_num + cluster_index + 1)
            ax.plot(i)

            topics = 0
            #plt.plot(i)
            #plt.show() #show orginal clusters
            #if len(cluster_assignments[index]) > 0:
            for cluster in cluster_assignments:
                if cluster == cluster_index:

                    ax.plot(assignments_data[topics])

                    # this stores the table details. Do topics than cluster
                    cell_text.append([topics,cluster_index])

                    topic_list.append([topics])


                topics +=1
            #plt.show()
            cluster_index +=1

        # Add a table at the bottom of the axes

        # this pushes the graphs to the left more
        PLT.subplots_adjust(right=0.7)

        # below plots the table
        #sets the posittion of the table
        left, width = 0.1, 0.6
        bottom, height = 0.1, 0.8
        left_table = left+width+0.1
        table_width = 0.15
        table_height = width/2.

        rect_table1 = [left_table, table_height+bottom , table_width, table_height]

        axTable1 = PLT.axes(rect_table1, frameon =False)
        axTable1.axes.get_yaxis().set_visible(False)

        axTable1.table(cellText=cell_text,
                              #rowLabels=rows,
                              #rowColours=colors,
                              colLabels=["topics","clusters"],
                              loc='upper center')

        print "Number of topics after each clustering"
        print len(topic_list)
        print len(assignments_data) # this is ok
        print cell_text

        PLT.show()
