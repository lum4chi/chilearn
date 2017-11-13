#!/usr/bin/env python3
# coding: utf-8
# author: Francesco Lumachi <francesco.lumachi@gmail.com>

import types
import numpy as np
from scipy.stats import f
from sklearn.utils import check_X_y, safe_sqr
from sklearn.base import clone


def f_classif(X, y):
    # TODO Ancora non ci siamo con la memoria (soprattutto le comprehension)
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


def _ranked_fit(self, X, y, step_score=None):
    """ This version add the capability to gather ranking to
    `sklearn.feature_selection.RFE` at each step.
    """
    # Parameter step_score controls the calculation of self.scores_
    # step_score is not exposed to users
    # and is used when implementing RFECV
    # self.scores_ will not be calculated when calling _fit through fit

    X, y = check_X_y(X, y, "csc")
    # Initialization
    n_features = X.shape[1]
    if self.n_features_to_select is None:
        n_features_to_select = n_features // 2
    else:
        n_features_to_select = self.n_features_to_select

    if 0.0 < self.step < 1.0:
        step = int(max(1, self.step * n_features))
    else:
        step = int(self.step)
    if step <= 0:
        raise ValueError("Step must be >0")

    support_ = np.ones(n_features, dtype=np.bool)
    ranking_ = np.ones(n_features, dtype=np.int)

    ### 1/3 - Declare ranking list ###
    grid_ranking_ = []
    ##################################

    if step_score:
        self.scores_ = []

    # Elimination
    while np.sum(support_) > n_features_to_select:
        # Remaining features
        features = np.arange(n_features)[support_]

        # Rank the remaining features
        estimator = clone(self.estimator)
        if self.verbose > 0:
            print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X[:, features], y)

        # Get coefs
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" or "feature_importances_" '
                               'attributes')

        # Get ranks
        if coefs.ndim > 1:
            ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
        else:
            ranks = np.argsort(safe_sqr(coefs))

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)

        # Eliminate the worse features
        threshold = min(step, np.sum(support_) - n_features_to_select)

        # Compute step score on the previous selection iteration
        # because 'estimator' must use features
        # that have not been eliminated yet
        if step_score:
            self.scores_.append(step_score(estimator, features))
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1

        ### 2/3 - Append computed ranking ###
        grid_ranking_.append(ranking_.copy())
        #####################################

    # Set final attributes
    features = np.arange(n_features)[support_]
    self.estimator_ = clone(self.estimator)
    self.estimator_.fit(X[:, features], y)

    # Compute step score when only n_features_to_select features left
    if step_score:
        self.scores_.append(step_score(self.estimator_, features))
    self.n_features_ = support_.sum()
    self.support_ = support_
    self.ranking_ = ranking_

    ### 3/3 - Stack the rankings in a np.array ###
    self.grid_ranking_ = np.array(grid_ranking_)
    ##############################################

    return self

def ranking_observer(rfe):
    """ Decorate an RFE instance with _ranked_fit """
    rfe._fit = types.MethodType(_ranked_fit, rfe)
    return rfe

def merge_support(arrays, inplace=False):
    """ Merge multiple support mask from an estimators pipeline. """
    assert all([t==np.ndarray for t in map(type,arrays)]), "Can merge only numpy arrays."
    assert len(arrays)>1, "Can merge only more than 1 numpy array."
    if not inplace:
        arrays = list(map(np.copy, arrays))
    for this, next in reversed(list(zip(arrays, arrays[1:]))):
        this[this==True] = next
    return this
