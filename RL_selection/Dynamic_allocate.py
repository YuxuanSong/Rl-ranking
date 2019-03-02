import cPickle as pkl
import dns_method.utils
from dns_method.ads_models import *
import  numpy as np

Workspace = "make-ipinyou-data/"
Compaign = "all/"
Relative_rank = open('Newlogs/relativerank_baseipinyou','rb')

Modelfile = 'models/base_nds_1.pkl'
Compaign = "all/"

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
winningrate = 0.02

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

Testdata = utils.read_yzx_data(Testfile)

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
Logfile = "Newlogs/Constant_baseline_ipinyou_%f.txt" %(winningrate)
fout = open(Logfile,'w')

test_preds = model.run(model.y_prob, utils.csr_2_input(Testdata[0]))
clicks = np.array([0])
index = 0

Threhold = findTherold(1-winningrate,relative_ranking)
print Threhold

total_budget = int(Testlength*winningrate)

while index< Testlength and total_budget>=0:
        if test_preds[index] >= Threhold:
            total_budget = total_budget - 1
            clicks = clicks + Testdata[1][index]
            index = index + 1
        else:
            index = index + 1
        if index%1000 == 0 and index!=0:
            fout.write(str(Testlength-index)+'\t'+str(clicks[0])+'\t'+str(total_budget)+'\n')
            print str(Testlength-index)+'\t'+str(clicks[0])+'\t'+str(total_budget)

