import Orange
import random
from pyfann import libfann
import numpy as np

import math

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

    def __init__(self, name="NeuralNetwork", n_mid=10, learning_rate=0.9, max_iter=1000,
                 desired_error=0.001):

        self.name = name
        self.n_mid = n_mid
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.desired_error = desired_error
    
    def __call__(self,data,weight=0):

        #convert and normalize attribute data
        X = data.to_numpy()[0] 
        self.minv = X.min(axis=0)
        self.maxv = X.max(axis=0)
        X = (X - self.minv) / self.maxv

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
            cvals = [0]+[sum(cvals[0:i+1]) for i in xrange(len(cvals))]  

            for i in xrange(len(data)):
                for j in xrange(len(cvals)-1):
                    if cvals[j+1] - cvals[j] > 2:
                        Y[i, cvals[j] + int(data[i].get_classes()[j])] = 1.0
                    else:
                        Y[i, cvals[j]] = float(data[i].get_classes()[j])
        else:
            y = np.array([int(d.get_class()) for d in data])
            n_classes = len(data.domain.class_var.values)
            if n_classes > 2:
                Y = np.eye(n_classes)[y]
            else:
                Y = y[:,np.newaxis]
      
        #initialize neural network
        self.ann = libfann.neural_net()
        self.ann.create_standard_array((len(X[0]), self.n_mid, len(Y[0])))
        self.ann.set_activation_function_output(libfann.SIGMOID_STEPWISE)
        self.ann.set_activation_function_hidden(libfann.SIGMOID_STEPWISE)
        self.ann.set_learning_rate(self.learning_rate)
        #self.ann.set_training_algorithm(libfann.TRAIN_BATCH)
        #self.ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
        self.ann.set_training_algorithm(libfann.TRAIN_RPROP)
        #self.ann.set_training_algorithm(libfann.TRAIN_QUICKPROP)
  
        nn_data = libfann.training_data()
        nn_data.set_train_data(X, Y)

        self.ann.train_on_data(nn_data, self.max_iter, 100, self.desired_error)

        return FANNClassifier(classifier=self.classify, domain = data.domain)

    def classify(self, x):
        x = (x - self.minv) / self.maxv
        return self.ann.run(x)

class FANNClassifier():
    
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

    def __call__(self,example, result_type=Orange.core.GetValue):

        if not self.domain.class_vars: example = [example[i] for i in xrange(len(example)-1)]
        input = np.array([float(e) for e in example])

        results = self.classifier(input)

        mt_prob = []
        mt_value = []
          
        if self.domain.class_vars:
            cvals = [len(cv.values) if len(cv.values) > 2 else 1 for cv in self.domain.class_vars]
            cvals = [0] + [sum(cvals[0:i]) for i in xrange(1, len(cvals) + 1)]

            for cls in xrange(len(self.domain.class_vars)):
                if cvals[cls+1]-cvals[cls] > 2:
                    cprob = Orange.statistics.distribution.Discrete(results[cvals[cls]:cvals[cls+1]])
                    cprob.normalize()
                else:
                    r = results[cvals[cls]]
                    cprob = Orange.statistics.distribution.Discrete([1.0 - r, r])

                mt_prob.append(cprob)
                mt_value.append(Orange.data.Value(self.domain.class_vars[cls], cprob.values().index(max(cprob))))
        else:
            if len(results) > 1:
                cprob = Orange.statistics.distribution.Discrete(results)
                cprob.normalize()
            else:
                r = results[0]
                cprob = Orange.statistics.distribution.Discrete([1.0 - r, r])
            
            mt_prob = cprob
            mt_value = Orange.data.Value(self.domain.class_var, cprob.values().index(max(cprob)))

        if result_type == Orange.core.GetValue: return tuple(mt_value) if self.domain.class_vars else mt_value
        elif result_type == Orange.core.GetProbabilities: return tuple(mt_prob) if self.domain.class_vars else mt_prob
        else: 
            return [tuple(mt_value), tuple(mt_prob)] if self.domain.class_vars else [mt_value, mt_prob] 

if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    data = Orange.data.Table('wdbc')
    l1 = FANNLearner(n_mid=10, learning_rate=0.7, max_iter=2000, desired_error=0.01)
    res = Orange.evaluation.testing.cross_validation([l1],data, 3)
   
    scores = Orange.evaluation.scoring.CA(res)

    for i in range(len(scores)):
        print res.classifierNames[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
