﻿import Orange
import random
from pyfann.libfann import *
import numpy as np
import neural_common
import math

SYMMETRIC_FUNCTIONS = [SIGMOID_SYMMETRIC, SIGMOID_SYMMETRIC_STEPWISE, GAUSSIAN_SYMMETRIC,
                       ELLIOT_SYMMETRIC, LINEAR_PIECE_SYMMETRIC, SIN_SYMMETRIC, COS_SYMMETRIC]

class FANNLearner(Orange.classification.Learner):
    """
    Wrapper for FANN (Fast Arfificial Neural Network) library. The code is based on
    the wrappers found in Orange.classification.neural.
    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)

        if data is None:   
            return self
        else:
            self.__init__(**kwargs)
            return self(data,weight)

    _defaults = dict(name="NeuralNetworkFANN", n_mid=10, learning_rate=0.9, max_iter=1000,
                 desired_error=0.001, normalization=True, activation_function=SIGMOID_STEPWISE,
                 algorithm=TRAIN_INCREMENTAL, error_function=ERRORFUNC_LINEAR,
                 stop_function=STOPFUNC_MSE)

    def __init__(self, **kwargs):

        # Raise exception if any non-supported keywords supplied
        if set(kwargs.keys()) - set(self._defaults.keys()):
            raise KeyError("unsupported keyword argument")
 
        # Update our instance with defaults, then keyword args
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
   
        self.is_symmetric = self.activation_function in SYMMETRIC_FUNCTIONS
    
    def __call__(self,data,weight=0):

        #convert and normalize attribute data
        X = data.to_numpy()[0]
        if self.normalization:
            self.minv = X.min(axis=0)
            self.maxv = X.max(axis=0)
            X = self.normalize(X) 

        #converts multi-target or single-target classes to numpy
        if data.domain.class_vars:
            for cv in data.domain.class_vars:
                if cv.var_type == Orange.feature.Continuous:
                    raise ValueError("non-discrete classes not supported")
        else:
            if data.domain.class_var.var_type == Orange.feature.Continuous:
                raise ValueError("non-discrete classes not supported")

        if data.domain.class_vars:
            cvals = [len(cv.values) if len(cv.values) > 2 else 1 for cv in data.domain.class_vars]
            Y = np.zeros((len(data), sum(cvals)))
            if self.is_symmetric:
                Y.fill(-1.0)
            cvals = [0]+[sum(cvals[0:i+1]) for i in xrange(len(cvals))]  

            for i in xrange(len(data)):
                for j in xrange(len(cvals)-1):
                    if cvals[j+1] - cvals[j] > 2:
                        Y[i, cvals[j] + int(data[i].get_classes()[j])] = 1.0
                    else:
                        if self.is_symmetric:
                            Y[i, cvals[j]] = float(data[i].get_classes()[j])*2-1
                        else:    
                            Y[i, cvals[j]] = float(data[i].get_classes()[j])
        else:
            if self.is_symmetric:
                y = np.array([float(d.get_class())*2-1 for d in data])
            else:
                y = np.array([float(d.get_class()) for d in data])
            n_classes = len(data.domain.class_var.values)
            if n_classes > 2:
                Y = np.eye(n_classes)[y]
            else:
                Y = y[:,np.newaxis]
      
        #initialize neural network
        self.ann = neural_net()
        self.ann.create_standard_array((len(X[0]), self.n_mid, len(Y[0])))
        self.ann.set_activation_function_output(self.activation_function)
        self.ann.set_activation_function_hidden(self.activation_function)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_training_algorithm(self.algorithm)
        self.ann.set_train_stop_function(self.stop_function)
        self.ann.set_train_error_function(self.error_function)
  
        nn_data = training_data()
        
        nn_data.set_train_data(X, Y)

        self.ann.train_on_data(nn_data, self.max_iter, 100, self.desired_error)

        return FANNClassifier(classifier=self.classify, domain = data.domain)

    def normalize(self, x):
       if self.is_symmetric: 
           return (x - self.minv) / (self.maxv*0.5) - 1.0
       else:
           return (x - self.minv) / self.maxv
       
    def classify(self, x):
        if self.normalization:
            x = self.normalize(x)
        res = self.ann.run(x)
        if self.is_symmetric:
            res = [(r + 1.0)/2.0 for r in res]
        return res

FANNClassifier = neural_common.NeuralNetClassifier

if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    data = Orange.data.Table('wdbc')
    l1 = FANNLearner(n_mid=10, learning_rate=0.7, max_iter=2000, desired_error=0.001)
                     #activation_function=SIGMOID_SYMMETRIC_STEPWISE)
    res = Orange.evaluation.testing.cross_validation([l1],data, 3)
   
    scores = Orange.evaluation.scoring.CA(res)

    for i in range(len(scores)):
        print res.classifierNames[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
