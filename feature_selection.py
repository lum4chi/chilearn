#!/usr/bin/env python3
# coding: utf-8
# author: Francesco Lumachi <francesco.lumachi@gmail.com>

import numpy as np
from scipy.stats import f
from datetime import datetime

# TODO Ancora non ci siamo con la memoria (soprattutto le comprehension)

def f_classif(X, y):
    groups, mask, counts = np.unique(y, return_inverse=True, return_counts=True)
    # SStotal
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"SStotal...")
    gmeans_ = np.array([X[mask==g,:].mean(axis=0) for g in range(len(groups))])
    means_ = gmeans_.mean(axis=0)
    sst_ = ((X - means_)**2).sum(axis=0)
    # SSwithin
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"SSwithin...")
    grouped_ss = [((X[mask==g,:] - gmeans_[g])**2).sum(axis=0) for g in range(len(groups))]
    #grouped_ss = [np.square(X[mask==g,:] - gmeans_[g]).sum(axis=0) for g in range(len(groups))]
    ssw_ = np.array(grouped_ss).sum(axis=0)
    # SSbetween
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"SSbetween...")
    grouped_ss = [(counts[g] * ((gmeans_[g] - means_)**2)) for g in range(len(groups))]
    #grouped_ss = [(counts[g] * np.square(gmeans_[g] - means_)) for g in range(len(groups))]
    ssb_ = np.array(grouped_ss).sum(axis=0)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"Completing...")
    # DF (degree of freedom)
    k, N = len(groups), X.shape[0]
    DFbetween, DFwithin, DFtotal = k - 1, N - k, N - 1
    # F-score
    MSbetween = ssb_/DFbetween
    MSwithin = ssw_/DFwithin
    F = MSbetween/MSwithin
    # p-value
    pval = f.sf(F, DFbetween, DFwithin)

    return F, pval
