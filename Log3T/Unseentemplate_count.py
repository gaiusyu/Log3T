# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime

import Log3T
import Log3T_random
import sys
import pandas as pd
sys.path.append('../../')


print('Hello')

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'threshold': 4
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 2
    },
    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'threshold': 5
    },
    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'threshold': 5
    },
    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'threshold': 10
    },
    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'threshold': 5
    },





}



bechmark_result = []

PC=[]
sum=0
PA_list=list()
count=0
for dataset, setting in benchmark_settings.items():

    ground_truth_read=pd.read_csv('../logs/'+dataset+'/'+dataset+'_2k.log_structured.csv')
    ground_truth=ground_truth_read['EventTemplate']
    ground_truth_list=list(ground_truth)
    count_list=list()
    unseen=0
    lines=0
    print(dataset)
    unseen_list=list()
    for template in ground_truth_list:
        template=' '.join(template)
        if template not in count_list:
            unseen_list.append(template)
        lines+=1
        if lines % 200 == 0:
            count_list=count_list+unseen_list
            print(str(len(unseen_list)))
            unseen_list=list()