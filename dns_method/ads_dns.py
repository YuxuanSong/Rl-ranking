import numpy as np
import utils as ut
import multiprocessing
from ads_models import *
from scipy.sparse import coo_matrix, vstack , csr_matrix
from tqdm import *
import random

from sklearn.metrics import roc_auc_score
cores = multiprocessing.cpu_count()

"""workdir and campaign  defined here(bebi/  make-ipinyou-data/ yoyi/)"""
workdir = 'bebi/'
# workdir = 'make-ipinyou-data/'
# workdir = 'yoyi/'
Compaign = '3/'
Testfile = workdir + Compaign+'test.yzx.txt'
Clicksfile = workdir + Compaign+ 'clicks_train'
Nonclicksfile = workdir + Compaign+'noclicks_train'

"""DNS ratio, i.e. top 1 of K sampling"""
DNS_k = 5
"""make pos and neg in a constant ratio"""
sample_ratio = 1

"""USED MODEL(LR, FM)"""
Used_model = "LR"
EXP = "MODEL" #model or baseline

if workdir == 'make-ipinyou-data/':
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
    elif Compaign =='3386/':
        input_dim = 556952
    elif Compaign == '2997/':
        input_dim =133541
    else:
        print 'notipinyou'
elif workdir == "bebi/":
    if Compaign == '3/':
        input_dim = 493072
    elif Compaign =='134848/':
        input_dim = 407153
    elif Compaign == '135059/':
        input_dim =531166
    elif Compaign == 'all/':
        input_dim =277077
    else:
        print 'not bebi'
else:
    input_dim = 7885617


#nds_ratio = utils.get_ndsratio(Trainfile,1)
#Landscape,ecpc,Max_price = utils.get_landscape(Trainfile)
name_field = utils.NAME_FIELD
clicks_data = utils.read_data(Clicksfile)
nonclicks_data = utils.read_data_ratio(Nonclicksfile,0.5)
test_data = utils.read_data(Testfile)

clicks_size = clicks_data[0].shape[0]
nonclicks_size = nonclicks_data[0].shape[0]
test_size = test_data[0].shape[0]

# utils.normilize_prob(Landscape)

def uniform_nds(ratio):
    """uniform sampling"""
    samples = int(ratio*clicks_size)
    index_list = []
    for i in range(samples):
        index_list.append(int(random.random()*nonclicks_size))
    index_list = np.array(index_list)
    return nonclicks_data[0][index_list],nonclicks_data[1][index_list]

def dns_sample():
    """multiprocess sampling function"""
    index_list = []
    for num in range(DNS_k):
        index_list.append(int(nonclicks_size*random.random()))
    return index_list

def generate_samples(model):
    """DNS sampling"""
    pool = multiprocessing.Pool(processes=cores)
    data = pool.map(dns_sample,range(int(sample_ratio*clicks_size)))
    tmp_traindata = clicks_data[0][:]
    tmp_label = clicks_data[1][:]
    pool.close()
    for list in data:
        sample_data = nonclicks_data[0][list]
        sample_preds = model.run(model.y_prob,ut.csr_2_input(sample_data))
        max_candidate = np.argmax(sample_preds) #have some questions about this
        Candi_data = nonclicks_data[0][[list[max_candidate]]]
        Candi_label = nonclicks_data[1][list[max_candidate]]
        tmp_traindata = vstack((tmp_traindata,Candi_data),format='csr')
        tmp_label=np.append(tmp_label,Candi_label)
    return tmp_traindata,tmp_label

def nds(Negative_file,ratios,nds_file):
    """uniform"""
    sample_numbers = int(ratios * clicks_size)
    fin = open(Negative_file,'r')
    fout = open(workdir +Compaign+nds_file,'w')
    for line in fin:
        if int((random.random()*nonclicks_size)) <= sample_numbers:
            fout.write(line)
    fin.close()
    fout.close()
    nds_set = utils.read_yzx_data(workdir +Compaign+nds_file)
    return  nds_set

# if we first let the pretraining happens
Pre_train_round = 500

num_round = 10
dns_round = 50
batch_size = 128
fout = open('Newlogs/dns_bebi3-dns5(nds_1.0,learn0.01)nonshuffle.log','w')
fout.write('loss    train-auc   test-auc\n')

#I will add the pretraining data to improve the training quality

def main(model):
    model_file = 'Newmodels/dns_bebi3-dns5(nds_1.0,learn0.01)nonshuffle.pkl'
    history_score = []
    for dns_iter in  tqdm(range(dns_round)):
        t_train= generate_samples(model)
        Train_size = t_train[0].shape[0]
        index = np.array(range(Train_size))
        np.random.shuffle(index)
        print index
        tmp_train = t_train[0],t_train[1]
        for round in tqdm(range(num_round)):
            fetches = [model.optimizer, model.loss]
            ls = []
            for j in range(tmp_train[0].shape[0]/batch_size+1):
                X_i, y_i = utils.slice(tmp_train, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
            #fetches = [model.optimizer, model.loss]
            train_preds = model.run(model.y_prob, ut.csr_2_input(tmp_train[0]))
            test_preds = model.run(model.y_prob, ut.csr_2_input(test_data[0]))
            train_score = roc_auc_score(tmp_train[1], train_preds)
            test_score = roc_auc_score(test_data[1], test_preds)
            print 'dns_iterations[%d]  round[%d]\tloss:%f\ttrain-auc: %f\teval-auc: %f' % (dns_iter,round, np.mean(np.array(ls)), train_score, test_score)
            fout.write( '%f\t%f\t%f\n' % (np.mean(np.array(ls)), train_score, test_score))
            fout.flush()
            history_score.append(test_score)
        model.dump(model_file)
    fout.close()

def uniform(model):
    model_file = 'Newmodels/dns_bebi3-dns5(nds_1.0,learn0.01)nonshuffle.pkl'
    history_score = []

    nds_ratio = 1.0
    Negtive_set = nds(Nonclicksfile,2,'allnds_2.0')
    print Negtive_set[1]
    tmp_train = (vstack((clicks_data[0],Negtive_set[0]),format='csr'),np.append(clicks_data[1],Negtive_set[1]))
    for round in tqdm(range(Pre_train_round)):
        fetches = [model.optimizer, model.loss]
        ls = []
        for j in range(tmp_train[0].shape[0] / batch_size + 1):
            X_i, y_i = utils.slice(tmp_train, j * batch_size, batch_size)
            _, l = model.run(fetches, X_i, y_i)
            ls.append(l)
        train_preds = model.run(model.y_prob, ut.csr_2_input(tmp_train[0]))
        test_preds = model.run(model.y_prob, ut.csr_2_input(test_data[0]))

        print tmp_train[1]
        train_score = roc_auc_score(tmp_train[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print 'Pre_round[%d]\tloss:%f\ttrain-auc: %f\teval-auc: %f' % (
        round, np.mean(np.array(ls)), train_score, test_score)
        fout.write('%f\t%f\t%f\n' % (np.mean(np.array(ls)), train_score, test_score))
        fout.flush()
        history_score.append(test_score)
    model.dump(model_file)
    fout.close()
if __name__ == '__main__':
    if Used_model == "LR":
        lr_params = {
            'input_dim': input_dim,
            'opt_algo': 'gd',
            'learning_rate': 0.01,
            'l2_weight': 0.0,
            'random_seed': 0
        }
        model = LR(**lr_params)
    else:
        fm_params = {
            'input_dim': input_dim,
            'factor_order': 2,
            'opt_algo': 'adam',
            'learning_rate': 0.01,
            'l2_w': 0.01,
            'l2_v': 0.001,
        }
        model = FM(**fm_params)
    if EXP == "MODEL":
        main(model)
    else:
        uniform(model)
