# Log3T
## Self-Supervised Log Parsing with Generalization Ability under Domain Shifts




### Reproduction


1. pip install -r requirements.txt

2. Run "Log3T/classify.py" to get the parsing accuracy on 16 benchmark datasets of Log3T directly.

3. Run "Log3T/parsing_time_training.py" to get the parsing accuracy with test-time-training at parsing time.

4. Run "Log3T/Unseentemplate_count.py" to get the number of unseen logs.

5. Run "Log3T/efficiency_evaluation.py" to reproduce results of efficiency evaluation (BGL dataset required)

The existing parser code reproduced in this paper relies on [LogPai](https://github.com/logpai).

For the parsers compared in our experiment, we reproduce the code in https://github.com/logpai/logparser.

paper link: https://arxiv.org/pdf/1811.03509.pdf.


