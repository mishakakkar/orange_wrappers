import numpy as np
import Orange

class NeuralNetClassifier(Orange.classification.Classifier):
    
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
        self.threshold = 0.5

    def set_threshold(self, threshold):
        """
        Sets the threshold between the classes. Only used for binary classification.
        Threshold should be in range [0,1].
        """
        self.threshold = threshold

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
                    mt_value.append(Orange.data.Value(self.domain.class_vars[cls], cprob.values().index(max(cprob))))
                else:
                    r = results[cvals[cls]]
                    cprob = Orange.statistics.distribution.Discrete([1.0 - r, r])
                    mt_value.append(Orange.data.Value(self.domain.class_vars[cls],
                                    1 if r > self.threshold else 0))

                mt_prob.append(cprob)
        else:
            if len(results) > 1:
                cprob = Orange.statistics.distribution.Discrete(results)
                cprob.normalize()
                mt_value = Orange.data.Value(self.domain.class_var, cprob.values().index(max(cprob)))
            else:
                r = results[0]
                cprob = Orange.statistics.distribution.Discrete([1.0 - r, r])
                mt_value = Orange.data.Value(self.domain.class_var, 1 if r > self.threshold else 0)
            
            mt_prob = cprob

        if result_type == Orange.core.GetValue: return tuple(mt_value) if self.domain.class_vars else mt_value
        elif result_type == Orange.core.GetProbabilities: return tuple(mt_prob) if self.domain.class_vars else mt_prob
        else: 
            return [tuple(mt_value), tuple(mt_prob)] if self.domain.class_vars else [mt_value, mt_prob] 
