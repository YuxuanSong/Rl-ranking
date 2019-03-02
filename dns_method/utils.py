import linecache
import cPickle
import  os
import heapq
import time
import multiprocessing
import cPickle as pkl
import numpy as np
import tensorflow as tf
from matrix import  *
from scipy.sparse import coo_matrix
from tqdm import *
import  random




def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_ndsratio(file,sampleratio):
    fin = open(file,'r')
    positive =0
    negative = 0
    for line in fin:
        record = line.split(' ')
        if int(record[0]) == 1:
            positive = positive + 1
        else:
            negative = negative + 1
    nds_r = positive * sampleratio*1.0/negative
    return  nds_r


def c_ctr(nds_ratio, pctr):
    # if pctr_file == 'dns_pctr.log':
    final_ctr = pctr
    # if mctr >= 0.4:
    #     final_ctr = 1
    # else:
    #     final_ctr = pctr
    c_pctr = final_ctr / (final_ctr + (1 - final_ctr) / nds_ratio)
    return c_pctr

def normilize_prob(Landscape):
    sum = 0
    for key in Landscape.keys():
        sum = sum + Landscape[key]

    for key in Landscape.keys():
        Landscape[key]= Landscape[key]/sum

def get_landscape(file_path):
    mp_dict = {}
    cost = 0
    clicks = 0
    Maxp = -1
    if not os.path.isfile(file_path):
        print "ERROR: file not exist. " + file_path
        exit(-1)
    fin = open(file_path,'r')
    for line in  fin:
        record = line.split(' ')
        mp = int(record[1])
        if int(record[0]) == 1:
            clicks= clicks +1
        cost = cost + mp

        if mp>Maxp:
            Maxp = mp
        if mp in mp_dict:
            mp_dict[mp] = mp_dict[mp]+1
        else:
            mp_dict[mp] = 1
    fin.close()
    ecpc = cost*1.0/clicks

    print "Landscape made"
    return mp_dict,ecpc,Maxp

# Get batch data from training set
def get_batch_data(file, index, size):  # 1,5->1,2,3,4,5
    user = []
    pos = []
    neg = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip()
        line = line.split()
        user.append(int(line[0]))
        pos.append(int(line[1]))
        neg.append(int(line[2]))

    user = np.array(user)
    pos = np.array(pos)
    neg = np.array(neg)
    return user, pos, neg


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def gen_auc_test(test_file, sess, generator):
    result = []
    user_item_test = cPickle.load(open(test_file))
    for u in user_item_test:
        ratings = sess.run(generator.user_item_rating, {generator.predict_uid: u})
        for item_pair in user_item_test[u]:
            pos = item_pair[0]
            neg = item_pair[1]
            if ratings[pos] > ratings[neg]:
                result.append(1)
            else:
                result.append(0)

    return np.mean(result)


def dis_auc_test(test_file, sess, discriminator):
    result = []
    user_item_test = cPickle.load(open(test_file))
    for u in user_item_test:
        ratings = sess.run(discriminator.u_items_rating, {discriminator.uid: u})
        for item_pair in user_item_test[u]:
            pos = item_pair[0]
            neg = item_pair[1]
            if ratings[pos] > ratings[neg]:
                result.append(1)
            else:
                result.append(0)

    return np.mean(result)


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.



DTYPE = tf.float32
FIELD_SIZES = [4, 25, 14, 131227, 43, 364, 5, 765, 996, 2, 2, 4, 2, 4, 2, 5]
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = 0
# Compaign ='all/'
# if Compaign == 'all/':
#     INPUT_DIM = 937750
# elif Compaign =='1458/':
#     INPUT_DIM = 560870
# elif Compaign =='2259/':
#     INPUT_DIM = 97500
# elif Compaign == '2261/':
#     INPUT_DIM =333223
# elif Compaign == '2821/':
#     INPUT_DIM = 460925
# elif Compaign =='3386/':
#     INPUT_DIM = 556952
# elif Compaign == '2997/':
#     INPUT_DIM =133541
# else:
#     print 'notipinyou'

# Compaign ='3/'
# if Compaign == '3/':
#     INPUT_DIM = 493072
# elif Compaign =='134848/':
#     INPUT_DIM = 407153
# elif Compaign =='135059/':
#     INPUT_DIM = 531166
# else:
#     print 'notbebi'
Compaign ='3/'
if Compaign == '3/':
    INPUT_DIM = 493072
elif Compaign =='134848/':
    INPUT_DIM = 407153
elif Compaign == '135059/':
    INPUT_DIM =531166
elif Compaign == 'all/':
    INPUT_DIM =277077
else:
    print 'not bebi'

# Compaign ='yoyi/'
# if Compaign == 'yoyi/':
#     INPUT_DIM = 7885617


OUTPUT_DIM = 1

NAME_FIELD = {'weekday': 0, 'hour': 1, 'useragent': 2, 'IP': 3, 'region': 4, 'city': 5, 'adexchange': 6, 'domain': 7,
              'slotid': 8, 'slotwidth': 9, 'slotheight': 10, 'slotvisibility': 11, 'slotformat': 12, 'creative': 13,
              'advertiser': 14, 'slotprice': 15}

STDDEV = 1e-3
MINVAL = -1e-2
MAXVAL = 1e-2


def partit_data(file_name):
    X = []
    y = []
    cnt = 1
    with open(file_name) as fin:
        part = 1
        for line in tqdm(fin):
            if cnt < 2500000:
                fields = line.strip().split()
                y_i = int(fields[0])
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                y.append(y_i)
                X.append(X_i)
                cnt += 1
            else:
                fields = line.strip().split()
                y_i = int(fields[0])
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                y.append(y_i)
                X.append(X_i)
                X = np.array(X)
                y = np.reshape(np.array(y), (-1, 1))
                X = libsvm_2_coo(X, (X.shape[0], INPUT_DIM)).tocsr()
                result = X, y
                pkl.dump(result, open(file_name + '%d' % part,'wb'))
                part = part +1
                X = []
                y = []
                cnt = 0
    if len(X)>0:
        X = np.array(X)
        y = np.reshape(np.array(y), (-1, 1))
        X = libsvm_2_coo(X, (X.shape[0], INPUT_DIM)).tocsr()
        result = X, y
        #output = open(file_name + '%d' % part, 'wb')
        pkl.dump(result,  open(file_name + '%d' % part, 'wb'))




def read_data(file_name):
    X = []
    y = []
    with open(file_name) as fin:
        cnt = 0
        for line in tqdm(fin):
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
            y.append(y_i)
            X.append(X_i)
            cnt += 1

    X = np.array(X)
    #X -= 1
    #exit(0)
    y = np.reshape(np.array(y), (-1, 1))
    X = libsvm_2_coo(X, (X.shape[0], INPUT_DIM)).tocsr()
    return X, y

def read_data_ratio(file_name,ratio):
    X = []
    y = []
    with open(file_name) as fin:
        cnt = 0
        for line in tqdm(fin):
            num = random.random()
            if num<ratio:
                fields = line.strip().split()
                y_i = int(fields[0])
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                y.append(y_i)
                X.append(X_i)
                cnt += 1
    X = np.array(X)
    #X -= 1
    #exit(0)
    y = np.reshape(np.array(y), (-1, 1))
    X = libsvm_2_coo(X, (X.shape[0], INPUT_DIM)).tocsr()
    return X, y

def read_yzx_data(file_name):
    X = []
    y = []
    z = []
    with open(file_name) as fin:
        for line in tqdm(fin):
            fields = line.strip().split()
            y_i = int(fields[0])
            z_i = int(fields[1])
            X_i = map(lambda x: int(x.split(':')[0]), fields[2:])
            z.append(z_i)
            y.append(y_i)
            X.append(X_i)
    X = np.array(X)
    #X -= 1
    #exit(0)
    y = np.reshape(np.array(y), (-1, 1))
    z = np.reshape(np.array(z), (-1,1))
    X = libsvm_2_coo(X, (X.shape[0], INPUT_DIM)).tocsr()
    return X, y, z

def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]

def load_bigfile(file):
    return pkl.load(open(file))


def libsvm_2_coo(libsvm_data, shape):
    #coo_rows = np.zeros_like(libsvm_data)

    index_value  = map(lambda x:np.ones_like(range(len(x))), libsvm_data[:])
    #index_value=np.ones_like(range(len(libsvm_data)))                           #format as index:1
    lib_matrix = libsvm_matrix(indices=libsvm_data,values=index_value,shape=shape)
    return lib_matrix.tocoo()



def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs

def Random_choice(csr_data,index_list):
    Data_chosen =csr_data[0][index_list[0]]
    Labels = csr_data[1][index_list[1]]
    for index in range(0,len(index_list)):
        if index != 0:
            Data_chosen.append()



def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels


def split_data(data):
    fields = []
    for i in range(len(FIELD_OFFSETS) - 1):
        start_ind = FIELD_OFFSETS[i]
        end_ind = FIELD_OFFSETS[i + 1]
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, FIELD_OFFSETS[-1]:])
    return fields, data[1]


def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print 'load variable map from', init_path, load_var_map.keys()
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'load':
            var_map[var_name] = tf.Variable(load_var_map[var_name])
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method])
            else:
                print 'BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape
        else:
            print 'BadParam: init method', init_method
    return var_map


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + \
               indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * shape[3] + indices[:, 3]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def max_pool_2d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat(1, [r1, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_2d(params, indices), [-1, k])


def max_pool_3d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat(1, [r1, r2, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


def max_pool_4d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
    r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
    r2 = tf.tile(r2, [1, shape[2] * k])
    r3 = tf.tile(r3, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    r3 = tf.tile(tf.reshape(r3, [-1, 1]), [shape[0] * shape[1], 1])
    indices = tf.concat(1, [r1, r2, r3, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_4d(params, indices), [-1, shape[1], shape[2], k])
