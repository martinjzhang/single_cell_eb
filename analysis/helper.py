## from https://github.com/bagavi/Meddit/blob/master/scripts/helper.py

import collections
import numpy as np
import scipy.sparse as sp_sparse
import tables, h5py
import pickle, time
import logging
import os,binascii, datetime
from sklearn.metrics.pairwise import pairwise_distances

# from tensorflow.examples.tutorials.mnist import input_data
logging.basicConfig(level=logging.DEBUG,format='(%(threadName)-10s) %(message)s',)

GeneBCMatrix = collections.namedtuple('GeneBCMatrix', ['gene_ids', 'gene_names', 'barcodes', 'matrix'])

"""
    Returns stacked (csc) sparse matrices all the files in the filepath.
    h5 files are hierarchically stored dataset.
"""
def get_matrix_from_h5_filepath(path, genome):
    filename_array = os.listdir(path)
    all_matrix = []
    for i, filename in enumerate(filename_array):
        dsets = {}
        with tables.open_file(path+filename, 'r') as f:
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
            temp_mat = sp_sparse.csc_matrix((dsets['data'], dsets['indices'], dsets['indptr']), shape=dsets['shape'])
            all_matrix += [temp_mat]
    final_ans = sp_sparse.hstack(all_matrix, format='csc')
    return final_ans


"""
    Returns a csc sparse matrix. This function is from 10x
"""
def get_matrix_from_h5(filename, genome, return_collection=False):
    with tables.open_file(filename, 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
            matrix = sp_sparse.csc_matrix((dsets['data'], dsets['indices'], dsets['indptr']), shape=dsets['shape'])
            if return_collection:
                return GeneBCMatrix(dsets['genes'], dsets['gene_names'], dsets['barcodes'], matrix)
            return matrix
        
        except tables.NoSuchNodeError:
            raise Exception("Genome %s does not exist in this file." % genome)
        except KeyError:
            raise Exception("File is missing one or more required datasets.")



"""
    Normalizes the gene_expressions along different cells
"""
def normalise(mat):
    d, n = mat.shape
    norms = np.array(1./mat.sum(axis=0))
    normaliser = sp_sparse.csc_matrix((norms.reshape(n,),(np.arange(n),np.arange(n))), shape=(n,n))
    return sp_sparse.csc_matrix.dot(mat,normaliser).transpose()

"""
l2 distance
"""
def l2_dist(X1,X2):
    return pairwise_distances(X1, X2, metric='l2', n_jobs=1)

"""
l1 distance
"""
def l1_dist(X1,X2):
    return pairwise_distances(X1, X2, metric='l1', n_jobs=1)

"""
Cosine distance
"""
def cosine_dist(X1, X2):
    return pairwise_distances(X1, X2, metric='cosine', n_jobs=1)