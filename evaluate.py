import datetime
from Log3T import Log3T
from Log3T import preprocess
import pandas as pd
from evaluator import evaluator
import sys
from Transfomer_encoder import transfomer_encoder
sys.path.append('../../')


model=transfomer_encoder.BERT()

benchmark_settings = {

    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'threshold': 4,
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'threshold': 5,
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'threshold': 3,
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'threshold': 5,
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 4,
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'threshold': 3,
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'threshold': 2,
    },
    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'threshold': 4,
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'threshold': 8,
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 1,
    },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'threshold': 2,
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'threshold': 4,
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'threshold': 3,
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'threshold': 6,
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'threshold': 4,
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'threshold': 8,
    },

}


bechmark_result = []
GA=list()
PC=[]
sum_ED=0
sum=0
GA_list=list()
editdistance_list=list()
count=0
lastlist=[]

for dataset, setting in benchmark_settings.items():
    start=datetime.datetime.now()
    parse = preprocess.format_log(
        log_format=setting['log_format'],
        indir='logs/'+dataset)
    form = parse.format(dataset+'_2k.log')
    ground_truth_read=pd.read_csv('logs/'+dataset+'/'+dataset+'_2k.log_structured_corrected.csv')
    ground_truth=ground_truth_read['EventTemplate']
    ground_truth_list=list(ground_truth)
    content = form['Content']
    template=pd.DataFrame()
    sentences = content.tolist()
    print(dataset)
    log_data,log_sentence=Log3T.log_to_model(sentences,stage='parse',regx=[],regx_use=False,dataset=dataset,variablelist=[]) # 如果你想使用过滤器，可以在regx=""处添加,并且将regx_use改为True,过滤器的设计可以根据错误解析的结果进行设定
    log_group,group_with_template,predict_label=Log3T.parse(log_data=log_data,modelpath='torch_model/model'+dataset,log_sentence=log_sentence, threshold=setting['threshold'],log_group={},logid=0,model=model)
    # lastlist=Log3T.variable_wordlist_generate(lastlist=[],log_sentences=log_sentence,ground_truth=ground_truth_list)
    # Log3T.writefile(filename='Variableset/variablelist'+dataset+'.csv', content=lastlist)
    end = datetime.datetime.now()
    print('time taken == #'+str(end-start))
    PA=evaluator.get_GA(group_with_template,ground_truth_list,ground_truth,sum=2000)
    print(dataset+'#GA = '+str(PA))
    print('\n')
    Event_id=0
    template_=list()
    template_id=list()
    template_num=list()
    for key in group_with_template.keys():
        list_mem=group_with_template[key]
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
    form['EventId']=log_sentence
    form.to_csv('Result/'+dataset+'result.csv')
    template.to_csv('Result/' + dataset + 'template.csv')
    edit_distance_mean,_=evaluator.get_editdistance(ground_truth_read,form)
    editdistance_list.append(edit_distance_mean)
    GA_list.append(dataset)
    GA_list.append(str(PA))
    sum+=PA
    sum_ED+=edit_distance_mean
    count+=1
    F1_measure, accuracy = evaluator.evaluate(
                           groundtruth='logs/'+dataset+'/'+dataset+'_2k.log_structured_corrected.csv',
                           parsedresult='Result/'+dataset+'result.csv'
                           )
    bechmark_result.append([dataset, F1_measure, PA, edit_distance_mean])
lastlist=set(lastlist)
print(' '.join(GA_list))
print('average parsing accuracy =  ' + str(sum / count))
print('\n=== Overall evaluation results ===')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy', 'Edit_distance_mean'])
df_result.set_index('Dataset', inplace=True)
print(df_result)
print('Average GA = '+str(sum / count))
print('Average ED = '+str(sum_ED / count))