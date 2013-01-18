import Orange
from pycudann import *
import numpy as np

import math

class CudaNNLearner(Orange.classification.Learner):
    """
    Wrapper for libcudann library. The code is based on the wrappers found in
    Orange.classification.neural
    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)

        if data is None:   
            return self
        else:
            self.__init__(**kwargs)
            return self(data,weight)

    def __init__(self, name="NeuralNetworkCuda", normalization=True, hidden_layers=None,
                 activation_functions=None, **kwargs):

        self.name = name
        self.normalization = normalization
        self.train_params = kwargs
  
        if hidden_layers is None:
            hidden_layers = [10] 
        if activation_functions is None:
            activation_functions = [SIGM, SIGM, SIGM]
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions

        self.is_symmetric = (self.activation_functions[0] == TANH)
    
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
        self.ann = FeedForwardNN([len(X[0])] + self.hidden_layers + [len(Y[0])],
                                 self.activation_functions)
        nn_data = LearningSet(X, Y)
        trainer = FeedForwardNNTrainer()
        trainer.selectNet(self.ann)
        trainer.selectTrainingSet(nn_data)
        trainer.train(**self.train_params)

        return CudaNNClassifier(classifier=self.classify, domain = data.domain)

    def normalize(self, x):
       if self.is_symmetric: 
           return (x - self.minv) / (self.maxv*0.5) - 1.0
       else:
           return (x - self.minv) / self.maxv
       
    def classify(self, x):
        if self.normalization:
            x = self.normalize(x)
        res = self.ann.compute(x)
        if self.is_symmetric:
            res = [(r + 1.0)/2.0 for r in res]
        return res

class CudaNNClassifier():
    
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
    l1 = CudaNNLearner(print_type=PRINT_ALL,max_epochs=2000)
    res = Orange.evaluation.testing.cross_validation([l1],data, 3)
   
    scores = Orange.evaluation.scoring.CA(res)

    for i in range(len(scores)):
        print res.classifierNames[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
