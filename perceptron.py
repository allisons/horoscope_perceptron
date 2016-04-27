#!/usr/bin/env python
#Copyright (c) 2016 Allison Sliter
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import division

from random import Random
from functools import partial
from operator import itemgetter
from collections import defaultdict


class LazyWeight(object):

    """
    Helper class for `MulticlassAveragedPerceptron`:

    Instances of this class are essentially triplets of values which
    represent a weight of a single feature in an averaged perceptron.
    This representation permits "averaging" to be done implicitly, and
    allows us to take advantage of sparsity in the feature space.
    First, as the name suggests, the `summed_weight` variable is lazily
    evaluated (i.e., computed only when needed). This summed weight is the
    one used in actual inference: we need not average explicitly. Lazy
    evaluation requires us to store two other numbers. First, we store the
    current weight, and the last time this weight was updated. When we
    need the real value of the summed weight (for inference), we "freshen"
    the summed weight by adding to it the product of the real weight and
    the time elapsed.

    # initialize
    >>> t = 0
    >>> lw = LazyWeight(t=t)
    >>> t += 1
    >>> lw.update(t, 1)
    >>> t += 1
    >>> lw.get()
    1

    # some time passes...
    >>> t += 1
    >>> lw.get()
    1

    # weight is now changed
    >>> lw.update(-1, t)
    >>> t += 3
    >>> lw.update(-1, t)
    >>> t += 3
    >>> lw.get()
    -1
    """

    def __init__(self, default_factory=int, t=0):
        self.timestamp = t
        self.weight = default_factory()
        self.summed_weight = default_factory()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__dict__)

    def get(self):
        """
        Return current weight
        """
        return self.weight

    def _freshen(self, t):
        """
        Apply queued updates, and update the timestamp
        """
        self.summed_weight += (t - self.timestamp) * self.weight
        self.timestamp = t

    def update(self, value, t):
        """
        Bring sum of weights up to date, then add `value` to the weight
        """
        self._freshen(t)
        self.weight += value

    def average(self, t):
        """
        Set `self.weight` to the summed value, for final inference
        """
        self._freshen(t)
        self.weight = self.summed_weight / t


class MulticlassAveragedPerceptron(object):
    """
    Multiclass classification via the averaged perceptron. Features
    are assumed to be binary, hashable (e.g., strings), and very sparse. 
    Labels must also be hashable.
    """

    def __init__(self, default=None, seed=None):
        self.classes = {default}
        self.random = Random(seed)
        self.weights = defaultdict(partial(defaultdict, LazyWeight))
        self.time = 0
    
    def fit(self, Y, Phi, epochs, alpha=1):
        # copy data so we can mutate it in place
        data = list(zip(Y, Phi))
        for _ in xrange(epochs):
            self.random.shuffle(data)
            for (y, phi) in data:
                self.fit_one(y, phi)
        self.finalize()

    def fit_one(self, y, phi, alpha=1):
        self.classes.add(y)
        yhat = self.predict(phi)
        if yhat != y:
            self.update(y, yhat, phi, alpha)
        
        
        
    def update(self, y, yhat, phi, alpha=1):
        """
        Given feature vector `phi`, reward correct observation `y` and
        punish incorrect hypothesis `yhat`, assuming that `y != yhat`.
        `alpha` is the learning rate (usually 1).
        """
        for feature in phi:
            labels = self.weights[feature]
            labels[y].update(+alpha, self.time)
            labels[yhat].update(-1*alpha, self.time)
        self.time += 1

    def predict(self, phi):
        """
        scores = dict()
        for y in self.classes:
            scores[y] = 0
        for feature in phi:
            labels = self.weights[feature]
            for (k, v) in labels.iteritems():
                scores[k] += v.get()
        (yhat, _) = max(scores.iteritems(), key=itemgetter(1))
        return yhat
        """
        scores = dict.fromkeys(self.classes, 0)
        for phi_i in phi:
            for (cls, weight) in self.weights[phi_i].iteritems():
                scores[cls] += weight.get()
        (yhat, _) = max(scores.iteritems(), key=itemgetter(1))
        return yhat        
                

    def finalize(self):
        """
        Prepare for inference by applying averaging

        TODO(kbg): also remove zero-valued weights?
        """
        for (phi_i, clsweights) in self.weights.iteritems():
            for (cls, weight) in clsweights.iteritems():
                weight.average(self.time)

