#!/usr/bin/env python3
# coding: utf-8
# author: Francesco Lumachi <francesco.lumachi@gmail.com>

import pandas as pd
import numpy as np
from itertools import islice
from sklearn.utils.validation import check_X_y


class KTopScoringPair:
    """ K-Top Scoring Pair classifier.
        This classifier evaluate maximum-likelihood estimation for P(X_i < X_i | Y),
        with X_i < X_i a pair of feature given a class Y. K determine how many pair
        evaluate. Then pairs are ranked by the primary score:
                            s = P(X_i < X_j | 0) - P(X_i < X_j | 1)
        Further detail can be found in [1].
        For its nature this is a binary classifier but it will not provide any error
        if found multiple label, score will be computed between first and second
        class. Multi-class classification can be achieved by using sklearn multiclass
        wrappers.

        Parameters
        ----------
        pairs : list of tuples with index of the feature to be considered.
            The feature will be tested in order, that is (X_i, X_j) will be counted
            for X_i < X_j.
        K : int. How many pairs will contribute to classification.
            It should be chosen as an odd int, to allow majority voting.
        t : int, optional (default=0)
            It can be used to adjust accuracy/specificity. By default it means that
                score_{ij} = (P(X_i < X_j | 0) - P(X_i < X_j | 1)) > t
        Attributes
        ----------
        estimated_proba_ : 2d array of float
            Estimated probability computed from training.
        rules_ : array of shape = [n_classes]
            Human-readable K rules found with training.
        ----------
        .. [1] AFSARI, Bahman, et al. Rank discriminants for predicting phenotypes
        from RNA expression. The Annals of Applied Statistics, 2014, 8.3: 1469-1491.
    """

    def __init__(self, pairs, K, t=0):
        self.pairs = pairs
        self.K = K
        self.t = t
        self._estimator_type = "classifier"
        # Defined after fitting
        self.estimated_proba_ = None
        self.rules_ = []
        self.classes_ = []

    def fit(self, X, y):
        """ Train the classifier.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            y : array-like of shape = [n_samples]
            Returns
            -------
            self : returns an instance of self.
        """
        X, y = check_X_y(X, y)  # Assert input is safe
        # Determine class and convert y accordingly
        self.classes_, y = np.unique(y, return_inverse=True)
        # Main statistics gathering
        Frequencies, Sizes = self._fit(X, y, self.pairs)
        # Compute likelihood probabilities
        self._compute_proba(Frequencies, Sizes)
        return self

    def _fit(self, X, y, pairs):
        # Instantiate dictionary as counter for (X_i, X_j) = |{X_i < X_i | Y}|
        pairs_dict = {l: dict() for l in range(len(self.classes_))}
        class_size = {l: 0 for l in range(len(self.classes_))}
        # Class loop
        for label in pairs_dict.keys():
            X_given_y = X[y==label]
            class_size[label] = X_given_y.shape[0]
            class_pairs = pairs_dict[label]
            # Pairs loop
            for X_i, X_j in pairs:
                class_pairs[(X_i, X_j)] = sum(X_given_y[:, X_i] < X_given_y[:, X_j])
        # Return statistics in a convenient format
        Freq, Size = pd.DataFrame(pairs_dict), pd.Series(class_size)
        return Freq, Size

    def predict(self, X, K=None, t=None):
        """ Predict the provided X.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            K : int, optional.
                Once estimated_proba_ were computed there is no problem to vary K and
                use K-rules different from __init__ time K
            t : int, optional
                Same as above
            Returns
            -------
            y : array-like of shape = [n_samples]
        """
        P = self.predict_proba(X, K)
        # Translate most probable class with its label
        return self.classes_[np.argmax(P, axis=1)]

    def predict_proba(self, X, K=None, t=None):
        """ Predict the provided X with probabilities.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            K : int, optional.
                Once estimated_proba_ were computed there is no problem to vary K and
                use K-rules different from __init__ time K
            t : int, optional
                Same as above
            Returns
            -------
            P : array of shape = [n_samples, n_class]
        """
        def vote_for(x):
            return [r['i<j'] if x[r['i']] < x[r['j']] else r['j<i'] for r in self.rules_]

        # Rebuild rules if K or t is different from __init__ time K
        if (K is not None and K != self.K) or (t is not None and t != self.t):
            P = self.estimated_proba_
            self.K = self.K if K is None else K
            self.t = self.t if t is None else t
            self.rules_ = self._scorer(P, self.K, self.t, P.columns[0], P.columns[1])

        # Gather votes for every sample -> V = (n, k)
        V = [vote_for(x) for _, x in X.iterrows()]
        # Group votes by class -> P (n, c)
        P = [{k: v for k, v in zip(*np.unique(v, return_counts=True))} for v in V]
        P = pd.DataFrame(P).fillna(0)
        # Normalized it to emit probabilities
        return (P / self.K).as_matrix()

    def partial_fit(self, X_batch, y_batch, classes):
        """ Train the classifier by chunk. This can take advantage of multiprocessing
            computation. Choose chunk dimension it is your discretion.
            Parameters
            ----------
            X_batch : iterator for an {array-like, sparse matrix} of
                shape = [n_samples, n_features]
            y_batch : iterator for an array-like of shape = [n_samples]
            classes : array-like, shape (n_classes,)
                Can't be inferred, then classes need to be passed as argument.
            Returns
            -------
            self : returns an instance of self.
        """
        from multiprocessing import Pool

        self.classes_ = np.array(sorted(classes))
        pool = Pool()
        # Process mapping (zip is needed because map can handle only one argument)
        Freq_chunks, Size_chunks = zip(*pool.map(self._chunk_worker, zip(X_batch, y_batch)))
        # Concatenate resultant dictionary for missing pairs, then group-by and
        # aggregate totals with a sum
        F, S = pd.concat(Freq_chunks), pd.concat(Size_chunks)
        Frequencies, Sizes = F.groupby(level=[0, 1]).sum(), S.groupby(S.index).sum()
        # Now statistics are complete, compute as normal fit
        self._compute_proba(Frequencies, Sizes)
        return self

    def _chunk_worker(self, X_y):
        # Assert input safely
        X, y = X_y
        X, y = check_X_y(X, y)
        # Translate y as label
        d = {k:v for k,v in zip(self.classes_, range(len(self.classes_)))}
        y = np.array(list(map(lambda x: d[x], y)))
        # Count frequencies-sizes for this chunk
        return self._fit(X, y, self.pairs)

    def _scorer(self, P, K, t, minus, plus):
        # Not efficient friendly, but produce human-readable rules.
        def formatted_rule(i, j, isPositive, score):
            if isPositive:
                return {"i":i, "j":j, "i<j":minus, "j<i":plus, "score":score}
            else:
                return {"i":i, "j":j, "i<j":plus, "j<i":minus, "score":score}

        # +/- scores depends on what is subtracted from what
        scores = P[minus] - P[plus]
        ranked = scores.abs().sort_values(ascending=False)
        # Compute rules, ranked by descending score
        rules = [formatted_rule(k[0], k[1], scores[k] > t, scores[k])
                 for k in islice(iter(ranked.keys()), K)]
        return rules

    def _compute_proba(self, Frequencies, Sizes):
        # Mainly for debugging purposes
        self.frequencies_, self.sizes_ = Frequencies, Sizes
        # Compute P = |{X_i < X_i | Y}| / |Y|
        P = Frequencies / Sizes
        self.estimated_proba_ = P
        # Build rules
        self.rules_ = self._scorer(P, self.K, self.t, P.columns[0], P.columns[1])

    def get_params(self, deep=True):
        return {"pairs": self.pairs, "K": self.K, "t": self.t}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def human_rules(self, features):
        """ Allow rules convertion for human reading.
            Parameters
            ----------
            features : list of feature name corresponding to i,j indexing
            Returns
            -------
            hr_rules : list of rules, with label converted according to input
        """
        import copy as cp

        hr_rules = cp.deepcopy(self.rules_)
        for d in hr_rules:
            d['i'], d['j'] = features[d['i']], features[d['j']]
            d['i<j'], d['j<i'] = self.classes_[d['i<j']], self.classes_[d['j<i']]
        return hr_rules
