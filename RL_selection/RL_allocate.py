import cPickle as pkl
import dns_method.utils
import numpy as np
from dns_method.ads_models import *

# Workspace = "make-ipinyou-data/"
# Compaign = "all/"
Relative_rank = open('Newlogs/relativerank_bebi3dns5','rb')

Modelfile = 'Newmodels/dns_bebi_3-1-5(nds_1.0,learn0.01)nonshuffle.pkl'
DNS_k = 5


# Workspace = 'yoyi/'
# Compaign = ''

Workspace = 'bebi/'
Compaign = '3/'

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


# if Workspace == 'yoyi/':
#     input_dim = 7885617
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

def GetRelativeranking(scores,List):
    Length = len(List)
    relative_rank = 1
    for i in range(Length):
        if scores>List[i]:
            return relative_rank
        else:
            relative_rank = relative_rank - 1.0/Length
    return  relative_rank

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
    clicks = np.array([0])
    table = GetDP(RoundAuctions, wr)
    Logfile = "Newlogs/DP_bebi3_budgetspend_dns5100000_perround%f.txt" % (wr)
    fout = open(Logfile, 'w')
    total_budget = int(100000 * wr)
    this_round_spend = 0
    while index < 100000:
        tmp_auctions = 100000
        tmp_budget = int(tmp_auctions * wr)
        while tmp_auctions > 0:
            if tmp_budget > 0:
                rx = GetRelativeranking(test_preds[index], relative_ranking)
                if rx + table[tmp_auctions - 1][tmp_budget - 1] > table[tmp_auctions - 1][tmp_budget]:
                    tmp_budget = tmp_budget - 1
                    total_budget  = total_budget - 1
                    tmp_auctions = tmp_auctions - 1
                    clicks = clicks + Testdata[1][index]
                    this_round_spend = this_round_spend + 1
                else:
                    tmp_auctions = tmp_auctions - 1
                index = index + 1
            else:
                tmp_auctions = tmp_auctions - 1
                index = index + 1
            if index >= Testlength:
                break
            if index % 2000 == 0 and index != 0:
                fout.write(str(this_round_spend) +' '+str(total_budget)+'\n')
                this_round_spend = 0
                # fout.write(str(Testlength - index) + '\t' + str(clicks[0]) + '\t' + str(total_budget) + '\n')
                # print str(Testlength - index) + '\t' + str(clicks[0]) + '\t' + str(total_budget)

    # print clicks,winningrate, int(Testlength*winningrate)-total_budget,index,total_budget
    print  "winning_rate dataset algrithm impressions auctions Total_auctions budgets clicks CTR"
    # print wr, "bebi3", "DP_lr_100000", int(Testlength * wr) - total_budget, index, Testlength, int(
    #     Testlength * wr), clicks[0], clicks[0] * 1.0 / (int(Testlength * wr) - total_budget),100000


# while index< Testlength:
#     tmp_auctions = 10000
#     tmp_budget = int(tmp_auctions*winningrate)
#     while tmp_auctions>0:
#         if tmp_budget >0:
#             rx = GetRelativeranking(test_preds[index],relative_ranking)
#             if rx+table[tmp_auctions-1][tmp_budget-1]>table[tmp_auctions-1][tmp_budget]:
#                 tmp_budget = tmp_budget-1
#                 total_budget = total_budget - 1
#                 tmp_auctions = tmp_auctions-1
#                 clicks= clicks + Testdata[1][index]
#             else:
#                 tmp_auctions = tmp_auctions - 1
#             index = index + 1
#         else:
#             tmp_auctions = tmp_auctions -1
#             index = index + 1
#         if index >= Testlength:
#             break
#         if index%1001 == 0 and index!=0:
#             fout.write(str(Testlength-index)+'\t'+str(clicks[0])+'\t'+str(total_budget)+'\n')
#             print str(Testlength-index)+'\t'+str(clicks[0])+'\t'+str(total_budget)
#
#
# # print clicks,winningrate, int(Testlength*winningrate)-total_budget,index,total_budget
# print  "winning_rate dataset algrithm impressions auctions Total_auctions budgets clicks CTR"
# print winningrate,"ipinyou","DP_dns5",int(Testlength*winningrate)-total_budget,index,Testlength,int(Testlength*winningrate),clicks[0],clicks[0]*1.0/(int(Testlength*winningrate)-total_budget)

Wr_list = [0.02]
# [0.004,0.008,0.012,0.016,0.02]
for i in Wr_list:
    Replay(i)




