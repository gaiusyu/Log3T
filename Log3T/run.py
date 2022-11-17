# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import Log3T
import Log3T_random
import sys
sys.path.append('../../')


print('Hello')
benchmark_settings = {

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
    },






}


bechmark_result = []

PC=[]
for dataset, setting in benchmark_settings.items():
    parse = Log3T.format_log(
        log_format=setting['log_format'],
        indir='../logs/'+dataset)
    form = parse.format(dataset+'_2k.log')

    content = form['Content']
    # logID = form['LineId']
    # Date = form['Date']
    # Time = form['Time']
    arr = content.to_numpy()
    sentences = arr.tolist()
    batch,log_sentence=Log3T.sentence_process(sentences,stage='train')
    Log3T.train(data='../Train_data/Training_data_BGL.csv',epoch_n=10,output=dataset,weight=1,TTT=False)
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