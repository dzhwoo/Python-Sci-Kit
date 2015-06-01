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

        ax = fig.add_subplot(base_subplot_num + i)
        ax.plot(data[i])

    PLT.show()