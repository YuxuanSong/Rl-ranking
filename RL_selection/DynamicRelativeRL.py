import cPickle as pkl
import dns_method.utils
import numpy as np
from dns_method.ads_models import *


def GetRelativeranking(scores,List):
    Length = len(List)
    relative_rank = 1
    for i in range(Length):
        if scores>List[i]:
            return relative_rank
        else:
            relative_rank = relative_rank - 1.0/Length
    return  relative_rank

def Updatelist(tmp_list):
    tmp_list[::-1].sort()
    length = len(tmp_list)
    step = length/100
    Test_ranklist = []
    for i in  range(step):
        Test_ranklist.append(tmp_list[i*100])

    return  Test_ranklist

C_parameter = 0.01
Update_fre = 100000

# Workspace = "make-ipinyou-data/"
# Compaign = "all/"
Relative_rank = open('Newlogs/relativerank_bebi3dns5','rb')

Modelfile = 'Newmodels/dns_bebi_3-1-5(nds_1.0,learn0.01)nonshuffle.pkl'
DNS_k = 5


Workspace = 'bebi/'
Compaign = '3/'

if Compaign == '3/':
    input_dim = 493072
elif Compaign =='134848/':
    input_dim = 407153
elif Compaign == '135059/':
    input_dim =531166
else:
    print 'not bebi'


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


RoundAuctions = 100000
#winningrate = 0.02

B,N = 1000,10000

def GetDP(auctions, wrate):
    Budgets = int(wrate*RoundAuctions)
    v = [[0 for i in range(Budgets + 1)] for j in range(auctions+1)]
    for i in range(1,auctions+1):
        for j in range(1,Budgets+1):
            v[i][j]= (1.0+(v[i-1][j]-v[i-1][j-1])*(v[i-1][j]-v[i-1][j-1]))/2+v[i-1][j-1]
    return v


relative_ranking = pkl.load(Relative_rank)

Testfile = Workspace+Compaign+"test.yzx.txt"

Testdata = utils.read_data(Testfile)

Testlength = Testdata[0].shape[0]
lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0,
        'init_path': Modelfile
}

model = My_LR(**lr_params)


test_preds = model.run(model.y_prob, utils.csr_2_input(Testdata[0]))

# clicks = np.array([0])
# index = 0


def Replay(wr):
    index = 0
    current_test = 0
    clicks = np.array([0])
    table = GetDP(RoundAuctions, wr)
    Logfile = "Newlogs/DP_bebi3_dns5-dynamicrl100000-0.9_%f.txt" % (wr)
    fout = open(Logfile, 'w')
    total_budget = int(Testlength * wr)
    test_tmp = []
    test_rank = []
    while index < Testlength:
        tmp_auctions = 100000
        tmp_budget = int(tmp_auctions * wr)
        while tmp_auctions > 0:
            test_tmp.append(test_preds[index])
            current_test = current_test + 1
            if tmp_budget > 0:
                rtrain = GetRelativeranking(test_preds[index], relative_ranking)
                if len(test_rank) > 0:
                    rtest = GetRelativeranking(test_preds[index], test_rank)
                    rx = rtrain*C_parameter+rtest*(1-C_parameter)
                else:
                    rx = rtrain
                if rx + table[tmp_auctions - 1][tmp_budget - 1] > table[tmp_auctions - 1][tmp_budget]:
                    tmp_budget = tmp_budget - 1
                    total_budget = total_budget - 1
                    tmp_auctions = tmp_auctions - 1
                    clicks = clicks + Testdata[1][index]
                else:
                    tmp_auctions = tmp_auctions - 1
                index = index + 1
            else:
                tmp_auctions = tmp_auctions - 1
                index = index + 1
            if index >= Testlength:
                break
            if current_test == Update_fre:
                current_test = 0
                test_rank = Updatelist(test_tmp)
                print "test update"
            if index % 100000 == 0 and index != 0:
                fout.write(str(Testlength - index) + '\t' + str(clicks[0]) + '\t' + str(total_budget) + '\n')
                print str(Testlength - index) + '\t' + str(clicks[0]) + '\t' + str(total_budget)

    # print clicks,winningrate, int(Testlength*winningrate)-total_budget,index,total_budget
    print  "winning_rate dataset algrithm impressions auctions Total_auctions budgets clicks CTR"
    print wr, "bebi3", "DP_dns5_100000-0.9dynamicrelative", int(Testlength * wr) - total_budget, index, Testlength, int(
        Testlength * wr), clicks[0], clicks[0] * 1.0 / (int(Testlength * wr) - total_budget),100000



Wr_list = [0.01,0.02,0.04,0.08,0.10]

for i in Wr_list:
    Replay(i)









