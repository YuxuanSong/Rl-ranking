import Dataset as dt
import cPickle as pkl
import tensorflow as tf
import numpy as np
from ads_models import  *
import  utils
from tqdm import  *
#set the constants
workdir = 'bebi/3/'
testfile = workdir+'test.yzx.txt'
# Train_file = 'make-ipinyou-data/2259/train.yzx.txt'
#
# nds_ratio = utils.get_ndsratio(Train_file,1)
# Landscape,ecpc,Max_price = utils.get_landscape(Train_file)



#fout1 = open('base_nds1_pctr.log','w')
fout2 = open('Newlogs/dns_bebi_3-1-2(nds_1.0,learn0.001).pctr.log','w')
#read test date
Test_part = utils.read_data(testfile)
testsize = Test_part[0].shape[0]
#set model
# Compaign = '2259/'
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
Compaign = '3/'
if Compaign == '3/':
    input_dim = 493072
elif Compaign =='134848/':
    input_dim = 407153
elif Compaign == '135059/':
    input_dim =531166
else:
    print 'not bebi'
dns_model_file = 'Newmodels/dns_bebi_3-1-2(nds_1.0,learn0.001).pkl'
lr_params_dns = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.001,
        'l2_weight': 0,
        'random_seed': 0,
        'init_path': dns_model_file
    }
dnsmodel = My_LR(**lr_params_dns)

#get_pctr
#pctr = basemodel.run(basemodel.y_prob, utils.csr_2_input(Test_part[0]))
pctr_dns =dnsmodel.run(dnsmodel.y_prob,utils.csr_2_input(Test_part[0]))
# for p in pctr:
#     fout1.write('%f\n' %p)
# fout1.close()

for p in pctr_dns:
    fout2.write('%f\n' %p)
fout2.close()
