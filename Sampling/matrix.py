import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from __init__ import general_max, str_2_value, is_str


class libsvm_matrix:
    def __init__(self, indices=None, values=None, tuples=None, shape=None):
        if indices is None and values is None and tuple is not None:
            indices = map(lambda x: x[0], tuples)
            values = map(lambda x: x[1], tuples)
        self.indices = np.array(indices)
        self.values = np.array(values)
        if shape is not None:
            self.shape = shape
        else:
            size = len(indices)
            max_indices = map(general_max, indices)
            space = max(max_indices) + 1
            self.shape = (size, space)

    def __len__(self):
        if self.shape is None:
            return 0
        else:
            return self.shape[0]

    def __getitem__(self, item):
        return libsvm_matrix(indices=self.indices[item], values=self.values[item])

    def __str__(self):
        return '%s\n%s' % (str(self.indices), str(self.values))

    def __repr__(self):
        return 'libsvm matrix with shape %s' % str(self.shape)

    def tocoo(self):
        coo_rows = []
        coo_cols = []
        coo_data = []
        for i in range(len(self.indices)):
            coo_rows.extend([i] * len(self.indices[i]))
            coo_cols.extend(self.indices[i])
            coo_data.extend(self.values[i])
        return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=self.shape)

    def tocsr(self):
        return self.tocoo().tocsr()

    def tocsc(self):
        return self.tocoo().tocsc()

    def toarray(self):
        return self.tocoo().toarray()

    def tocompound(self, sub_spaces):
        return csc_2_compound(self.tocsc(), sub_spaces).tolibsvm()


class compound_matrix:
    def __init__(self, data_list=None, shape=None):
        for d_i in data_list:
            if get_mat_format(d_i) not in {'array', 'libsvm', 'csr', 'coo', 'csc'}:
                print 'unknown format'
                exit(0)

        self.data_list = data_list
        if shape is not None:
            self.shape = shape
        else:
            shape_list = []
            for d_i in data_list:
                shape_list.append(d_i.shape)
            shape_list = np.array(shape_list)
            self.shape = (np.max(shape_list[:, 0]), np.sum(shape_list[:, 1]))

    def __len__(self):
        if self.shape is None:
            return 0
        else:
            return self.shape[0]

    def __getitem__(self, item):
        return compound_matrix(map(lambda x: x[item], self.data_list))

    def __str__(self):
        return '\n'.join(map(str, self.data_list))

    def __repr__(self):
        return 'compound matrix with shape %s' % str(self.shape)

    def tocoo(self, merge=False):
        if not merge:
            return compound_matrix(map(lambda x: x.tocoo(), self.data_list))
        else:
            return self.tocsc(merge).tocoo()

    def tocsr(self, merge=False):
        if not merge:
            return compound_matrix(map(lambda x: x.tocsr(), self.data_list))
        else:
            return self.tocsc(merge).tocsr()

    def tocsc(self, merge=False):
        if not merge:
            return compound_matrix(map(lambda x: x.tocsc(), self.data_list))
        else:
            return hstack(map(lambda x: x.tocsc(), self.data_list))

    def toarray(self, merge=False):
        if not merge:
            return compound_matrix(map(lambda x: x.toarray(), self.data_list))
        else:
            return hstack(map(lambda x: x.toarray(), self.data_list))

    def toarray_or_csr(self):
        data_list = []
        for d_i in self.data_list:
            if get_mat_format(d_i) in {'array', 'csr'}:
                data_list.append(d_i)
            else:
                data_list.append(d_i.tocsr())
        return compound_matrix(self.data_list)

    def tolibsvm(self, merge=False):
        libsvm_data_list = []
        sub_spaces = []
        for i in range(len(self.data_list)):
            data_i = self.data_list[i]
            if get_mat_format(data_i) == 'csr':
                libsvm_data_i = csr_2_libsvm(data_i, True)
            elif get_mat_format(data_i) == 'coo':
                libsvm_data_i = coo_2_libsvm(data_i, True)
            elif get_mat_format(data_i) == 'csc':
                libsvm_data_i = csc_2_libsvm(data_i, True)
            elif get_mat_format(data_i) == 'array':
                libsvm_data_i = array_2_libsvm(data_i, True)
            else:
                libsvm_data_i = data_i
            libsvm_data_list.append(libsvm_data_i)
            sub_spaces.append(libsvm_data_i.shape[1])
        if not merge:
            return compound_matrix(libsvm_data_list)
        else:
            return csc_2_libsvm(self.tocsc(merge))


def str_2_array(line):
    assert is_str(line), 'only accept string'
    return map(str_2_value, line.strip().split())


def str_2_libsvm(line):
    assert is_str(line), 'only accept string'
    if ':' in line:
        fields = map(lambda x: x.split(':'), line.strip().split())
        ind = map(lambda x: str_2_value(x[0]), fields)
        val = map(lambda x: str_2_value(x[1]), fields)
        return ind, val
    else:
        ind = map(lambda x: str_2_value(x), line.strip().split())
        val = [1] * len(ind)
        return ind, val


def coo_2_libsvm(coo_mat, reorder=False):
    assert get_mat_format(coo_mat) == 'coo', 'only accept coo format'
    data = np.vstack((coo_mat.row, coo_mat.col, coo_mat.data)).transpose()
    if reorder:
        data = sorted(data, key=lambda x: (x[0], x[1]))
    indices = []
    values = []
    for i in range(len(data)):
        r, c, d = data[i]
        if len(indices) <= r:
            while len(indices) <= r:
                indices.append([])
                values.append([])
            indices[r].append(c)
            values[r].append(d)
        elif len(indices) == r + 1:
            indices[r].append(c)
            values[r].append(d)
    while len(indices) < coo_mat.shape[0]:
        indices.append([])
        values.append([])
    return libsvm_matrix(indices=np.array(indices), values=np.array(values), shape=coo_mat.shape)


def csr_2_libsvm(csr_mat, reorder=False):
    assert get_mat_format(csr_mat) == 'csr', 'only accept csr format'
    return coo_2_libsvm(csr_mat.tocoo(), reorder=reorder)


def csc_2_libsvm(csc_mat, reorder=False):
    assert get_mat_format(csc_mat) == 'csc', 'only accept csc format'
    return coo_2_libsvm(csc_mat.tocoo(), reorder=reorder)


def array_2_libsvm(arr, reorder=False):
    assert get_mat_format(arr) == 'array', 'only accept numpy array'
    return coo_2_libsvm(coo_matrix(arr), reorder=reorder)


def csc_2_compound(csc_mat, sub_spaces):
    assert get_mat_format(csc_mat) == 'csc', 'only accept csc format'
    data_list = []
    for i in range(len(sub_spaces)):
        offset = sum(sub_spaces[:i])
        data_list.append(csc_mat[:, offset:offset + sub_spaces[i]])
    return compound_matrix(data_list=data_list)


def array_2_compound(arr, sub_spaces):
    assert get_mat_format(arr) == 'array', 'only accept numpy array'
    data_list = []
    for i in range(len(sub_spaces)):
        offset = sum(sub_spaces[:i])
        data_list.append(arr[:, offset:offset + sub_spaces[i]])
    return compound_matrix(data_list=data_list)


def csr_2_compound(csr_mat, sub_spaces):
    assert get_mat_format(csr_mat) == 'csr', 'only accept csr format'
    return csc_2_compound(csr_mat.tocsc(), sub_spaces).tocsr()


def coo_2_compound(coo_mat, sub_spaces):
    assert get_mat_format(coo_mat) == 'coo', 'only accept coo format'
    return csc_2_compound(coo_mat.tocsc(), sub_spaces).tocoo()


def get_mat_format(mat):
    if isinstance(mat, np.ndarray):
        return 'array'
    elif isinstance(mat, csr_matrix):
        return 'csr'
    elif isinstance(mat, coo_matrix):
        return 'coo'
    elif isinstance(mat, csc_matrix):
        return 'csc'
    elif isinstance(mat, libsvm_matrix):
        return 'libsvm'
    elif isinstance(mat, compound_matrix):
        return 'compound'
    else:
        print 'unknown format'
        exit(0)


def as_mat_format(mat, fmt):
    if mat is None:
        return None
    elif get_mat_format(mat) == 'array':
        if fmt == 'array':
            return mat
        elif fmt == 'coo':
            return coo_matrix(mat)
        elif fmt == 'csr':
            return coo_matrix(mat).tocsr()
        elif fmt == 'csc':
            return coo_matrix(mat).tocsc()
        elif fmt == 'libsvm':
            return array_2_libsvm(mat, True)
        elif fmt == 'compound':
            raise NotImplementedError('not implemented')
    elif get_mat_format(mat) == 'coo':
        if fmt == 'array':
            return mat.toarray()
        elif fmt == 'coo':
            return mat
        elif fmt == 'csr':
            return mat.tocsr()
        elif fmt == 'csc':
            return mat.tocsc()
        elif fmt == 'libsvm':
            return coo_2_libsvm(mat, True)
        elif fmt == 'compound':
            raise NotImplementedError('not implemented')
    elif get_mat_format(mat) == 'csr':
        if fmt == 'array':
            return mat.toarray()
        elif fmt == 'coo':
            return mat.tocoo()
        elif fmt == 'csr':
            return mat
        elif fmt == 'csc':
            return mat.tocsc()
        elif fmt == 'libsvm':
            return csr_2_libsvm(mat, True)
        elif fmt == 'compound':
            raise NotImplementedError('not implemented')
    elif get_mat_format(mat) == 'csc':
        if fmt == 'array':
            return mat.toarray()
        elif fmt == 'coo':
            return mat.tocoo()
        elif fmt == 'csr':
            return mat.tocsr()
        elif fmt == 'csc':
            return mat
        elif fmt == 'libsvm':
            return csc_2_libsvm(mat, True)
        elif fmt == 'compound':
            raise NotImplementedError('not implemented')
    elif get_mat_format(mat) == 'libsvm':
        if fmt == 'array':
            return mat.toarray()
        elif fmt == 'coo':
            return mat.tocoo()
        elif fmt == 'csr':
            return mat.tocsr()
        elif fmt == 'csc':
            return mat.tocsc()
        elif fmt == 'libsvm':
            return mat
        elif fmt == 'compound':
            raise NotImplementedError('not implemented')
    elif get_mat_format(mat) == 'compound':
        if fmt == 'array':
            return mat.toarray()
        elif fmt == 'coo':
            return mat.tocoo()
        elif fmt == 'csr':
            return mat.tocsr()
        elif fmt == 'csc':
            return mat.tocsc()
        elif fmt == 'libsvm':
            return mat.tolibsvm()
        elif fmt == 'compound':
            return mat


def vstack(blocks):
    fmt = 'array'
    for b_i in blocks:
        if get_mat_format(b_i) != 'array':
            fmt = 'csr'
            break
    if fmt == 'array':
        return np.vstack(blocks)
    else:
        return vstack(map(lambda x: x.tocsr(), blocks))


def hstack(blocks):
    fmt = 'array'
    for b_i in blocks:
        if get_mat_format(b_i) != 'array':
            fmt = 'csc'
            break
    if fmt == 'array':
        return np.hstack(blocks)
    else:
        return hstack(map(lambda x: x.tocsc(), blocks))

# def coo_2_sparse_tensor(coo_mat):
#     indices = np.transpose(np.vstack((coo_mat.row, coo_mat.col)))
#     values = coo_mat.data
#     shape = coo_mat.shape
#     return indices, values, shape
#
#
# def mat_2_tensor(mat):
#     mat_fmt = get_mat_format(mat)
#     if mat_fmt == 'array':
#         return mat
#     elif mat_fmt in {'csr', 'csc', 'coo', 'libsvm'}:
#         return coo_2_sparse_tensor(mat.tocoo())
#     else:
#         print mat_fmt, 'not supported'
#         exit(0)