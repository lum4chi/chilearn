#!/usr/bin/env python3
# coding: utf-8
# author: Francesco Lumachi <francesco.lumachi@gmail.com>

import os
import h5py
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.sparse import lil_matrix
from itertools import combinations, zip_longest
from functools import partial
from multiprocessing import Pool
from datetime import datetime
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_selection.base import SelectorMixin


class H5Binarizer(BaseEstimator, TransformerMixin):
    """Binarize data (set feature values to 0 or 1) according to a threshold.

    Intended for working only with h5py dataset: therefore can't be used in
    GridSearchCV because an h5py.Dataset can't be pickled for multiprocessing.
    Use instead `sklearn.preprocessing.Binarizer` for in-memory computation.

    Parameters
    ----------
    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
    """

    def __init__(self, threshold=0.0, sparse=None, verbose=0):
        self.threshold = threshold
        self.verbose = verbose
        self.sparse = sparse

    def _binarize(self, C):
        t = self.threshold
        return np.greater(C, t).astype("int8")

    def _chunks_generator(self, X):
        # Compute chunks indexes
        m = X.shape[1]
        chunk_size = X.chunks[1]
        assert chunk_size != None, "This method works only with chunked dataset."
        starts = range(0, m, chunk_size)
        ends = range(chunk_size, m, chunk_size)
        indexes = zip_longest(starts, ends, fillvalue=m)
        return indexes # : [(chunk0_start, chunk0_end), (chunk1_start, ...]

    def fit(self, X, y=None):
        assert type(X)==h5py.Dataset, "This class works only with h5py.Dataset"
        return self

    def transform(self, X, y="deprecated", copy=None):
        assert type(X)==h5py.Dataset, "This class works only with h5py.Dataset"
        # Chunk-wise
        chunks = self._chunks_generator(X)

        # dok: No(memory), coo: No(no slicing), bsr: No(no slicing)
        # csc/csr: No(t incrementally efficient)
        if self.sparse:
            B = lil_matrix(X.shape, dtype="int8")
        else:  # Fortran-order preferred because column-wise fashion
            B = np.empty(X.shape, dtype="int8", order='F')

        for i, indexes in enumerate(chunks):
            start, end = indexes
            B[:,start:end] = self._binarize(X[:,start:end])
            if self.verbose:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "Chunk",i ,
                      "({},{})".format(start,end),"transformed.")

        apply_sparsity = {
            None: lambda B: B,
            "csc": lambda B: B.tocsc(copy=False),
            "csr": lambda B: B.tocsr(copy=False),
            "bsr": lambda B: B.tobsr(copy=False),
        }

        return apply_sparsity[self.sparse](B)


class LogitThreshold(BaseEstimator, SelectorMixin):

    def __init__(self, threshold=0., verbose=0):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        """Compute logit of X (axis=0).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        Returns
        -------
        self
        """
        n, m = X.shape

        if type(X)==h5py.Dataset and hasattr(X, "chunks")!= None:
            # by chunk
            chunk_m = X.chunks[1]
            chunks = pd.Series(np.arange(m)).groupby(np.arange(m) // chunk_m)
            E = np.zeros(m)
            for c, indexes in chunks:
                start = indexes.values[0]
                end = indexes.values[-1] + 1
                if self.verbose:
                    print("Computing chunk {}: ({},{})...".format(c,start,end))
                np.sum(X[:,start:end], axis=0, out=E[start:end])
        else:
            # in-memory
            E = np.sum(X, axis=0)   # Favourable cases

        # Logit computation
        P = E / n               # Ratio
        self.logit_ = logit(P)  # Logit(P/(1-P))

        return self

    def _get_support_mask(self):
        t = self.threshold
        L = self.logit_
        return (-t < L) & (L < t)
