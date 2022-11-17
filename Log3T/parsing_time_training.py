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
    start=datetime.datetime.now()
    parse = Log3T.format_log(
        log_format=setting['log_format'],
        indir='../logs/'+dataset)
    form = parse.format(dataset+'_2k.log')
    ground_truth_read=pd.read_csv('../logs/'+dataset+'/'+dataset+'_2k.log_structured.csv')
    ground_truth=ground_truth_read['EventTemplate']
    ground_truth_list=list(ground_truth)
    content = form['Content']
    # logID = form['LineId']
    # Date = form['Date']
    # Time = form['Time']
    arr = content.to_numpy()
    sentences = arr.tolist()
    lenth=len(sentences)
    initial=sentences[:int(0.1*lenth)]
    batch, log_sentence = Log3T.sentence_process(initial, stage='train')
    batch_c, log_sentence_c = Log3T.sentence_process(sentences, stage='classfy')
    Log3T.train(data='../Train_data/Training_data_BGL.csv', epoch_n=1, output=dataset, weight=1,TTT=False)
    began=0
    count=0
    PA_differstage=list()
    for log in sentences:
        if began == 0:
            template_group = {}
            began=1
        if count>2000:
            break
        template_group,new_group = Log3T.classfywords(data=batch_c[count:count+1], output=dataset, weight=1, modelpath='model' + dataset,
                                   log_sentence=log_sentence_c, threshold=setting['threshold'],template_group=template_group,logid=count,stage='one')
        if ((count+1) % int(0.1*lenth)) == 0 and count !=0:
            if count+1<2000:
                part = sentences[count+1:count+int(0.1*lenth)+1]
                batch_n, log_sentence_n = Log3T.sentence_process(part, stage='train')
                Log3T.train(data='../Train_data/Training_data_BGL.csv', epoch_n=1, output=dataset, weight=1,TTT=True)
            PA = Log3T.get_PA(new_group, ground_truth_list[count-int(0.1*lenth)+1:count+1], ground_truth[count-int(0.1*lenth)+1:count+1],sum=int(0.1*lenth))
            PA_differstage.append(str(PA))
            template_group = {}
            print('PA of last '+str(int(0.1*lenth))+'  =  ' + str(PA))
        count += 1
    print('parsing time training result == ###'+' '.join(PA_differstage))


'''
    PA,template,template_num,template_set=Log3T.(dataset, arr, setting['delimiter'])
    form['template']=template
    form['template_num']=template_num
    form.to_csv('../SaveFiles&Output/Parseresult/' + dataset + '/' + dataset + 'result.csv', index=False)
    with open('../SaveFiles&Output/Parseresult/' + dataset + '/' + dataset + 'templates.csv', 'w') as f:
        template_num = 0
        for k in template_set:
            f.write(' '.join(k))
            f.write('\n')
        f.close()
    lenth=len(form)

    PC.append(setting['log_file']+'     '+str(PA))
print('####### % Parsing Accuracy % ########'+'\n')
for pa_ in PC:
    print(str(pa_))
print('ok')
'''