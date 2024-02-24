import datetime
from Log3T import Log3T
import pandas as pd
from Transfomer_encoder import transfomer_encoder
from Log3T import preprocess
'''
model will be constantly update using test time training
model2 is used to load parameters trained with only first batch
model3 is used to load parameters trained with all logs

'''
model=transfomer_encoder.BERT()
model2=transfomer_encoder.BERT()
model3=transfomer_encoder.BERT()

benchmark_settings = {

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'threshold': 6,
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 1,
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 1,
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}', r'J([a-z]{2})'],
        'epoch': 1,
    },
}



for dataset, setting in benchmark_settings.items():
    print(dataset)
    start=datetime.datetime.now()
    fromated_log = preprocess.format_log(
        log_format=setting['log_format'],
        indir='logs/'+dataset)
    form = fromated_log.format(dataset+'_2k.log')
    log_content = form['Content'].to_numpy().tolist()
    ground_truth_read=pd.read_csv('logs/'+dataset+'/'+dataset+'_2k.log_structured_corrected.csv')
    ground_truth_list=list(ground_truth_read['EventTemplate'])
    variablelist=Log3T.read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
    parse_data,log_sentence=Log3T.log_to_model(log_content,stage='parse',regx=[],regx_use=False,dataset=dataset,variablelist=[]) # if you want to use regular expression filter, you should set regx_use as 'True'
    train_data,_=Log3T.log_to_model(log_content,stage='train',regx=setting['regex'],regx_use=False,
                                        dataset=dataset,variablelist=variablelist)
    PA_of_batches,constant,variable,PA_of_batches_withoutTTT,PA_of_batches_origin\
        =Log3T.online_parsing_withTTT(train_data=train_data,parse_data=parse_data,log_sentence=log_sentence,
                                      threshold=setting['threshold'], ground_truth_list=ground_truth_list,
                                      modelpath='torch_model/model' + dataset, model=model, model2=model2,
                                      model3=model3)
    end_time=datetime.datetime.now()
    print('Process time = '+str(end_time-start))
    print(dataset+' \nGA of batches (batch size = '+str(100)+') : ')
    for index,GA in enumerate(PA_of_batches):
        print('batch','%04d' % (index),'    WithTTT =', '{:.4f}'.format(GA),'    WithoutTTT =', '{:.4f}'.format(PA_of_batches_withoutTTT[index]),'    Trained with all logs =', '{:.4f}'.format(PA_of_batches_origin[index]), '    Improvement =', '+{:.4f}'.format(GA-PA_of_batches_withoutTTT[index]) if GA-PA_of_batches_withoutTTT[index] > 0 else '{:.4f}'.format(GA-PA_of_batches_withoutTTT[index]))
    print('average label of constant words = '+constant)
    print('average label of variable words = ' + variable)
