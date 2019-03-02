import Dataset as dt
import cPickle as pkl
import tensorflow as tf
import numpy as np
from dns_method.ads_models import  *
import  dns_method.utils
from tqdm import  *
#set the constants

#path########################
# workdir = 'make-ipinyou-data/'
# Compaign = 'all/'
# input_dim = 937750 #for all ipinyou
# if Compaign == 'all/':
#     input_dim = 937750
# elif Compaign =='1458/':
#     input_dim = 560870
# elif Compaign =='2259/':
#     input_dim = 97500
# elif Compaign == '2261/':
#     input_dim =333223
# elif Compaign == '2821/':
#     input_dim = 460925
# elif Compaign =='3386/':
#     input_dim = 556952
# elif Compaign == '2997/':
#     input_dim =133541
# else:
#     print 'notipinyou'
# workdir = 'bebi/'
# Compaign = '135059/'

workdir = 'bebi/'
Compaign = 'all/'

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


if workdir == 'yoyi/':
    input_dim = 7885617

#---------------------define the file to load--------


relative_rank_file = 'Newlogs/relativerank_base-bebiall_5.1'
Trainfile = workdir+Compaign+'train.yzx.txt'
#Modelfile = 'models/dns_all_1-10.pkl'
Modelfile = 'Newmodels/baseline_nonshuf_bebiall0.01_nds_15.1.pkl'
###############################

#hyperparameters############
lr_params_dns = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.001,
        'l2_weight': 0,
        'random_seed': 0,
        'init_path': Modelfile
    }
bucket_length = 5000
############################


#init variables##############
relative_rank = []
Model = My_LR(**lr_params_dns)
Traindata = utils.read_data_ratio(Trainfile,0.5)
Trainsize = Traindata[0].shape[0]

Total_length = Trainsize/bucket_length
#############################

predict_train = Model.run(Model.y_prob,utils.csr_2_input(Traindata[0]))

predict_train[::-1].sort()

for Inx in range(1,Total_length+1):
    if Inx != Total_length:
        relative_rank.append(predict_train[Inx*bucket_length])
    else:
        relative_rank.append(predict_train[Trainsize-1])

fout = open('reranking_base-bebiall','w')
for i in relative_rank:
    fout.write(str(i))
    fout.write('\n')

pkl.dump(relative_rank,open(relative_rank_file,'wb'))
###############################








