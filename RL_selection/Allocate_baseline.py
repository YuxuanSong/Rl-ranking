import cPickle as pkl
import utils
from dns_method.ads_models import *
import  numpy as np

# Workspace = "bebi/"
Compaign = "all/"
# Workspace  = 'yoyi/'
Workspace = 'make-ipinyou-data/'



#---------------------define the file to load--------
# Compaign = ''
Relative_rank = open('Newlogs/relativerank_bebi3dns5','rb')
##############
#constant catch
###############
Modelfile = 'Newmodels/dns_bebi_3-1-5(nds_1.0,learn0.01)nonshuffle.pkl'
# Compaign = "all/"
# if Workspace == 'yoyi/':
#     input_dim = 7885617
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
elif Compaign =='3386/':
    input_dim = 556952
elif Compaign == '2997/':
    input_dim =133541
else:
    print 'notipinyou'
# if Compaign == '3/':
#     input_dim = 493072
# elif Compaign =='134848/':
#     input_dim = 407153
# elif Compaign == '135059/':
#     input_dim =531166
# else:
#     print 'not bebi'



def GetRelativeranking(scores,List):
    Length = len(List)
    relative_rank = 1
    for i in range(Length):
        if scores>List[i]:
            return relative_rank
        else:
            relative_rank = relative_rank - 1.0/Length
    return  relative_rank

def findTherold(score,List):
    Length = len(List)
    Start = 1
    for i in range(Length):
        if Start>score:
            Start = Start - 1.0/Length
        else:
            return List[i]


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

def Replay(wr):
    Logfile = "Newlogs/Constant_dns5ipinyou_budgetspend_perround%f.txt" % (wr)
    fout = open(Logfile, 'w')

    test_preds = model.run(model.y_prob, utils.csr_2_input(Testdata[0]))
    clicks = np.array([0])
    index = 0

    Threhold = findTherold(1 - wr, relative_ranking)
    print Threhold

    total_budget = int(100000 * wr)
    this_round_spend = 0
    while index < 100000:
        if test_preds[index] >= Threhold  and total_budget >= 0:
            total_budget = total_budget - 1
            clicks = clicks + Testdata[1][index]
            index = index + 1
            this_round_spend = this_round_spend+1
        else:
            index = index + 1
        if index % 2000 == 0 and index != 0:
            fout.write(str(this_round_spend) +' '+str(total_budget)+'\n')
            this_round_spend = 0
            # fout.write(str(Testlength - index) + '\t' + str(clicks[0]) + '\t' + str(total_budget) + '\n')
            #print str(Testlength - index) + '\t' + str(clicks[0]) + '\t' + str(total_budget)
    fout.close()
    print  "winning_rate dataset algrithm impressions auctions Total_auctions budgets clicks CTR"
    # print wr, "bebi3", "conslr", int(Testlength * wr) - total_budget, index, Testlength, int(
    #     Testlength * wr), clicks[0], clicks[0] * 1.0 / (int(Testlength * wr) - total_budget)


Wr_list = [0.02]

for i in Wr_list:
    Replay(i)