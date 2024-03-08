import numpy as np
import scipy.sparse as sp
import torch


def adj_to_tensor(adj):
    r"""

    Description
    -----------
    Convert adjacency matrix in scipy sparse format to torch sparse tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    Returns
    -------
    adj_tensor : torch.Tensor
        Adjacency matrix in form of ``N * N`` sparse tensor.

    """

    if type(adj) != sp.coo.coo_matrix:
        adj = adj.tocoo()
    sparse_row = torch.LongTensor(adj.row).unsqueeze(1)
    sparse_col = torch.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(sparse_concat.t(), sparse_data, torch.Size(adj.shape))

    return adj_tensor


def adj_preprocess(adj, adj_norm_func=None, mask=None, model_type="torch", device='cpu'):
    r"""

    Description
    -----------
    Preprocess the adjacency matrix.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or a tuple
        Adjacency matrix in form of ``N * N`` sparse matrix.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``None``.
    mask : torch.Tensor, optional
        Mask of nodes in form of ``N * 1`` torch bool tensor. Default: ``None``.
    model_type : str, optional
        Type of model's backend, choose from ["torch", "cogdl", "dgl"]. Default: ``"torch"``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    adj : torch.Tensor or a tuple
        Adjacency matrix in form of ``N * N`` sparse tensor or a tuple.

    """

    if adj_norm_func is not None:
        adj = adj_norm_func(adj)
    if model_type == "torch":
        if type(adj) is tuple or type(adj) is list:
            if mask is not None:
                adj = [adj_to_tensor(adj_[mask][:, mask]).to(device)
                       if type(adj_) != torch.Tensor else adj_[mask][:, mask].to(device)
                       for adj_ in adj]
            else:
                adj = [adj_to_tensor(adj_).to(device)
                       if type(adj_) != torch.Tensor else adj_.to(device)
                       for adj_ in adj]
        else:
            if type(adj) != torch.Tensor:
                if mask is not None:
                    adj = adj_to_tensor(adj[mask][:, mask]).to(device)
                else:
                    adj = adj_to_tensor(adj).to(device)
            else:
                if mask is not None:
                    adj = adj[mask][:, mask].to(device)
                else:
                    adj = adj.to(device)
    elif model_type == "dgl":
        if type(adj) is tuple:
            if mask is not None:
                adj = [adj_[mask][:, mask] for adj_ in adj]
            else:
                adj = [adj_ for adj_ in adj]
        else:
            if mask is not None:
                adj = adj[mask][:, mask]
            else:
                adj = adj
    return adj


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)####
    idx_val = range(200, 500)
    idx_test = range(500, 1500)####

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
