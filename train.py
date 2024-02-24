from Log3T import Log3T,preprocess
import sys
from Transfomer_encoder import transfomer_encoder

sys.path.append('../../')


model=transfomer_encoder.BERT()

print('Hello')
benchmark_settings = {

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 7,
        'regex': [r'core\.\d+'],
        'epoch': 1,
    },


}


bechmark_result = []

PC=[]
for dataset, setting in benchmark_settings.items():
    parse = preprocess.format_log(
        log_format=setting['log_format'],
        indir='logs/'+dataset)
    form = parse.format(dataset+'_2k.log')
    content = form['Content']
    arr = content.to_numpy()
    sentences = arr.tolist()
    variablelist=Log3T.read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
    batch,log_sentence=Log3T.log_to_model(sentences,stage='train',regx=setting['regex'],regx_use=False,dataset=dataset,variablelist=variablelist)
    Log3T.train(batch,epoch_n=setting['epoch'],output=dataset,model=model)
    '''
  'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'threshold': 5,
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'epoch': 1,
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'threshold': 5,
        'regex': [r'(\d+\.){3}\d+'],
        'epoch': 1,
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'threshold': 5,
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 1,
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'threshold': 4,
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'epoch': 1,
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 7,
        'regex': [r'core\.\d+'],
        'epoch': 1,
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'threshold': 4,
        'regex': [r'=\d+'],
        'epoch': 1,
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'threshold': 4,
        'regex': [r'(\d+\.){3}\d+'],
        'epoch': 1,
    },
    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'threshold': 7,
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'epoch': 1,
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'threshold': 6,
        'regex': [r'0x.*?\s'],
        'epoch': 1,
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 2,
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}', r'J([a-z]{2})'],
        'epoch': 1,
    },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'threshold': 15,
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'epoch': 2,
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'threshold': 7,
        'regex': [],
        'epoch': 2,
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'threshold': 3,
        'regex': [r'(\d+\.){3}\d+'],
        'epoch': 1,
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'threshold': 3,
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 1,
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'threshold': 6,
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'epoch': 1,
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'threshold': 10,
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 2,
    },
'''