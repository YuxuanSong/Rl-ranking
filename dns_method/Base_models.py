import tensorflow as tf
import cPickle
import numpy as np
import random
import utils as ut
import time
import multiprocessing
from ads_models import *
from scipy.sparse import coo_matrix, vstack , csr_matrix
from tqdm import *
from sklearn.metrics import roc_auc_score
cores = multiprocessing.cpu_count()

workdir = 'make-ipinyou-data/'
Compaign = 'all/'
# workdir = 'bebi/'
#workdir = 'make-ipinyou-data/'
# workdir = 'yoyi/'


#---------------------define the file to load--------
# Compaign = ''
Testfile = workdir + Compaign+'test.yzx.txt'
Clicksfile = workdir + Compaign+ 'clicks_train'
Nonclicksfile = workdir+Compaign + 'noclicks_train'

# input_dim= utils.INPUT_DIM
input_dim = 937750 #for all ipinyou
if Compaign == 'all/':
    input_dim = 937750
elif Compaign =='1458/':
    input_dim = 560870
elif Compaign =='2259/':
    input_dim = 97500
elif Compaign == '2261/':
    input_dim =333223
elif Compaign == '2821/':
    input_dim = 460925
elif Compaign == '3386/' :
    input_dim = 556952
elif Compaign == '2997/':
    input_dim =133541
else:
    print 'notipinyou'#workdir = 'make-ipinyou-data/'

# if Compaign == '3/':
#     input_dim = 493072
# elif Compaign =='134848/':
#     input_dim = 407153
# elif Compaign == '135059/':
#     input_dim =531166
# elif Compaign == 'all/':
#     input_dim =277077
# else:
#     print 'not bebi'

# if workdir == 'yoyi/':
#     input_dim = 7885617


Testfile = workdir +Compaign+ 'test.yzx.txt'
Nonclick_file = workdir + Compaign+'noclicks_train'
Click_file = workdir +Compaign+'clicks_train'


name_field = utils.NAME_FIELD


test_data = utils.read_data(Testfile)
click_data = utils.read_data(Click_file)
Nonclick_data = utils.read_data_ratio(Nonclick_file,0.5)

test_size = test_data[0].shape[0]
positive_size = click_data[0].shape[0]
num_round = 201
batch_size = 256


def cout_lines(file):
    lines = 0
    fin = open(file,'r')
    for line in fin:
        lines = lines + 1
    fin.close()
    return lines

Negative_size = cout_lines(Nonclick_file)

def nds(Negative_file,ratios,nds_file):
    sample_numbers = int(ratios * positive_size)
    fin = open(Negative_file,'r')
    fout = open(workdir +Compaign+nds_file,'w')
    for line in fin:
        if int((random.random()*Negative_size)) <= sample_numbers:
            fout.write(line)
    fin.close()
    fout.close()

    nds_set = utils.read_data(workdir +Compaign+nds_file)
    return  nds_set
#--------------------------------
Logfile = 'Newlogs/baseline_nonshuf_'+'bebiall_learningrate_0.01'+'_ratio1.log'
fout = open(Logfile,'w')
def main(model):
    history_score = []
    nds_ratio = 1
    Negtive_set = nds(Nonclick_file,nds_ratio,'bebiall_nonshuffle_nds_1.0')
    model_file = 'Newmodels/baseline_nonshuf_bebiall0.01_nds_%d5.1.pkl' %(int(nds_ratio))
    tmp_train = (vstack((click_data[0],Negtive_set[0]),format='csr'),np.append(click_data[1],Negtive_set[1]))
    for round in tqdm(range(num_round)):
        fetches = [model.optimizer, model.loss]
        ls = []
        for j in range(tmp_train[0].shape[0]/batch_size+1):
            X_i, y_i = utils.slice(tmp_train, j * batch_size, batch_size)
            _, l = model.run(fetches, X_i, y_i)
            ls.append(l)
        fetches = [model.optimizer, model.loss]
        train_preds = model.run(model.y_prob, ut.csr_2_input(tmp_train[0]))
        test_preds = model.run(model.y_prob, ut.csr_2_input(test_data[0]))
        if round%10 == 0:
             model.dump(model_file)
        train_score = roc_auc_score(tmp_train[1], train_preds)
        test_score = roc_auc_score(np.reshape(test_data[1], [-1]), test_preds)
        print 'round[%d]\tloss:%f\ttrain-auc:%f\teval-auc: %f' %(round, np.mean(np.array(ls)),train_score, test_score)
        fout.write('%f\t%f\t%f\n' % (np.mean(np.array(ls)), train_score,test_score))
        fout.flush()
        history_score.append(test_score)
    fout.close()

if __name__ == '__main__':
    lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0
    }
    model = LR(**lr_params)
    main(model)
