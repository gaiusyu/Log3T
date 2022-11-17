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

    'BGL': {
        'log_file': 'BGL/BGL.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 4
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
    form = parse.format(dataset+'.log')
    ground_truth_read=pd.read_csv('../logs/'+dataset+'/'+dataset+'_2k.log_structured.csv')
    ground_truth=ground_truth_read['EventTemplate']
    ground_truth_list=list(ground_truth)
    content = form['Content']
    # logID = form['LineId']
    # Date = form['Date']
    # Time = form['Time']
    arr = content.to_numpy()
    sentences = arr.tolist()
    batch,log_sentence=Log3T.sentence_process(sentences,stage='classfy')
    Group,new_group,predict_label=Log3T.classfywords(data=batch,output=dataset,weight=1,modelpath='model'+dataset+'9',log_sentence=log_sentence, threshold=setting['threshold'],template_group={},logid=0,stage='two')
    end = datetime.datetime.now()

print(' Time cost = ## '+str(end-start))


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