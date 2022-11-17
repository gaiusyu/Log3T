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
'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'threshold': 2
    },
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'threshold': 4
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'threshold': 5
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'threshold': 6
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'threshold': 5
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 4
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'threshold': 3
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'threshold': 8
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'threshold': 3
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 2
    },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'threshold': 5
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'threshold': 5
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'threshold': 3
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'threshold': 3
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'threshold': 4,
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'threshold': 10
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
    template=pd.DataFrame()
    # logID = form['LineId']
    # Date = form['Date']
    # Time = form['Time']
    arr = content.to_numpy()
    sentences = arr.tolist()
    batch,log_sentence=Log3T.sentence_process(sentences,stage='classfy',regx='') # 如果你想使用过滤器，可以在regx=""处添加
    Group,new_group,predict_label=Log3T.classfywords(data=batch,output=dataset,weight=1,modelpath='model'+dataset+'9',log_sentence=log_sentence, threshold=setting['threshold'],template_group={},logid=0,stage='two')


    line=0
    label=list()
    name=["image","<*>",'at', "(<*>)):", "checking"]
    for log in ground_truth_list:
        if "checking" in log:
            label.append(predict_label[line])
        line+=1
    label_csv = pd.DataFrame(columns=name,data=label)
    label_csv.to_csv('../label_analysis/' + dataset + 'label.csv')

    end = datetime.datetime.now()
    PA=Log3T.get_PA(new_group,ground_truth_list,ground_truth,sum=2000)
    print('time taken == #'+str(end-start))
    print('\n')
    Event_id=0
    template_=list()
    template_id=list()
    template_num=list()
    for key in new_group.keys():
        list_mem=new_group[key]
        template_.append(key)
        template_id.append('Event'+str(Event_id))
        template_num.append(str(len(list_mem)))
        for id in list_mem:
            sentences[id]=key
            log_sentence[id]='Event'+str(Event_id)
        Event_id+=1
    template['template']=template_
    template['template_id'] = template_id
    template['members_num'] = template_num
    form['template']=sentences
    form['template_id']=log_sentence
    form.to_csv('../Result/'+dataset+'result.csv')
    template.to_csv('../Result/' + dataset + 'template.csv')
    PA_list.append(dataset)
    PA_list.append(str(PA))
    sum+=PA
    count+=1
print(' '.join(PA_list))
print('average parsing accuracy =  '+str(sum / count))

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



 'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'threshold': 2
    },
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'threshold': 4
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'threshold': 5
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'threshold': 6
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'threshold': 5
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 4
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'threshold': 3
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'threshold': 8
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'threshold': 3
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 2
    },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'threshold': 5
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'threshold': 5
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'threshold': 3
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'threshold': 3
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'threshold': 4,
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'threshold': 10
    },

'''