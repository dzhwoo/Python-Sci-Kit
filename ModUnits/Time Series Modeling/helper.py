#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:      Store all helpers method here that we can reuse over time
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


def TakeFileConvertIntoDictionary(inputfile,keyindex,valueindex):

    output_dict={}
    for e, line in enumerate( open(inputfile) ):
        tmp_line = line.split(",")
        temp_topic = re.sub("[^a-zA-Z]","",tmp_line[keyindex])
        temp_topic = temp_topic.lower()
        output_dict[temp_topic] = tmp_line[valueindex]

    return output_dict

    print "Hello World"

##def TakeFileConvertIntoDictionary(inputfile,keyindex,valueindex,shouldClean):
##
##    output_dict={}
##    for e, line in enumerate( open(inputfile) ):
##        tmp_line = line.split(",")
##        if shouldClean == True:
##            temp_topic = re.sub("[^a-zA-Z]","",tmp_line[keyindex])
##            temp_topic = temp_topic.lower()
##        else:
##            temp_topic = tmp_line[keyindex]
##        output_dict[temp_topic] = tmp_line[valueindex]
##
##    return output_dict
##
##    print "Hello World"


def ConvertStringToDatetime(date_str,date_format):
    #return datetime.strptime(date_str, '%a %b %d %H:%M:%S +0000 %Y')
    return datetime.strptime(date_str, date_format)

def ImportFileConvertToNumpyArray(filepath,numskiprows,user_delimiter,user_dtype):
    return numpy.loadtxt(filepath,skiprows=numskiprows, delimiter = user_delimiter,dtype = user_dtype)

def ImportCSVFileConvertToNumpyArray(filepath):
    reader = csv.reader( open(filepath) )
    l = list(reader)
    a = numpy.array(l)
    return a

def WriteListToFile(inputlist,outfilepath):

    f=open(outfilepath, 'a')
    f.writelines("%s" % l for l in inputlist)
    f.flush()
    f.close()

def ReadFileIntoList(inputlist,colindex):

    outtext =[]

    for e, line in enumerate( open(inputlist) ):
        tmp_text = line.split(",")[colindex]
        tmp_text = tmp_text.replace("\"", "")
        tmp_text = tmp_text.replace("#", "")

        outtext.append(tmp_text)

    return outtext

def ReadFileIntoListWithSpecialFormat(inputlist,colindex):

    outtext =[]

    for e, line in enumerate( open(inputlist) ):
        tmp_text = line.split(",")[colindex]
        tmp_text = re.sub("[^a-zA-Z]","",tmp_text)
        tmp_text = tmp_text.lower()

        outtext.append(tmp_text)

    return outtext

def FilterFileBasedOnList(inputfilepath,colindex,inputlist,outputfile):

    outtext =[]

    for e, line in enumerate( open(inputfilepath) ):

        tmp_text = line.split(",")[colindex]
        tmp_text = tmp_text.replace("#", "")

        if tmp_text in inputlist:
            outtext.append(line)

    return outtext


# HELPER FUNCTIONS FOR STATISTICAL ANALYSIS : Sampling etc

def AddRandomNumberToFile(inputfilepath,outputfilepath,SamplingRatio):

    outlines = []

    #1 read inputfile line by line
    for e, line in enumerate( open(inputfilepath) ):
        line = line.rstrip("\n")

        rand_num = random.randint(1,SamplingRatio)

        if rand_num == SamplingRatio:
            #2 then append random number to add of line
            line = line + "," + str(rand_num) + "\n"

            outlines.append(line)

    #3 Once done then write out to file
    WriteListToFile(outlines,outputfilepath)


def PartitionFileIntoTrainingAndTestSet(inputfilepath,train_outputfilepath,test_outputfilepath,SamplingRatio):

    #TODO, may need to improve this later to specify sample size and then choose data.
    outlines_train = []
    outlines_test = []

    #1 read inputfile line by line
    for e, line in enumerate( open(inputfilepath) ):
        line = line.rstrip("\n")

        rand_num = random.randint(1,SamplingRatio)

        if rand_num == SamplingRatio:
            #2 then append random number to add of line
            line = line + "," + str(rand_num) + "\n"

            outlines_train.append(line)
        elif rand_num == SamplingRatio - 1:
            #2 then append random number to add of line
            line = line + "," + str(rand_num) + "\n"

            outlines_test.append(line)

    #3 Once done then write out to file
    WriteListToFile(outlines_train,train_outputfilepath)
    WriteListToFile(outlines_test,test_outputfilepath)

def FilterFile(inputListToFilter,inputFilepathToFilter,outputfilepath,inputfiletype,colFilterList,colInputFile):

    filterlist = []
    outputlist =[]

    for e, line in enumerate( open(inputListToFilter) ):
        temp_topic = line.split(",")[colFilterList].lower()
        #temp_topic = re.sub("[^a-zA-Z0-9]","",temp_topic)
        filterlist.append(temp_topic)

    #step3: go through transactions file and reduce
    reduced = 0
    checked = False # this controls if we do the remaining files
    index_count = 0

    #open tweets file
    for e, line in enumerate( open(inputFilepathToFilter) ):

        txt = line.split(",")[colInputFile]

        output =set(txt).intersection(filterlist)

        if len(output) >=1:
            outputlist.append(line)

        time_to_process_each_file = 2

        if index_count % 20000 == 0:
            #estimated_remaning_rows = total_files_per_day * 20000 - index_count
            #estimated_time_remaining = estimated_remaning_rows/total_rows_per_file * time_to_process_each_file
            print index_count
            break
            #print estimated_remaning_rows,estimated_time_remaining

        index_count +=1
  #print e, reduced, datetime.now() - start

    #write out trending tweets
    f=open(outputfilepath, 'a')
    f.writelines("%s" % l for l in outputlist)
    f.flush()
    f.close()




