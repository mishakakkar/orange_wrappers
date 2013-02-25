import Orange
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class GBCLearner(Orange.classification.Learner):
    """
    Wrapper for GradientBoostingClassifier from scikit-learn.
    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)

        if data is None:
            return self
        else:
            self.__init__(**kwargs)
            return self(data,weight)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self,data,weight=0):

        #convert and normalize attribute data
        XY = data.to_numpy()
        X = XY[0]
        Y = XY[1]

        gbc = GradientBoostingClassifier(**self.kwargs)
        gbc.fit(X, Y)

        return GBCClassifier(classifier=gbc, domain = data.domain)

class GBCClassifier(Orange.classification.Classifier):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, example, result_type=Orange.core.GetValue):
        input = np.array([float(example[i]) for i in xrange(len(example)-1)])
        
        if result_type == Orange.core.GetValue:
            return Orange.data.Value(self.domain.class_var, int(self.classifier.predict(input)[0]))
        elif result_type == Orange.core.GetProbabilities:
            return self.classifier.predict_proba(input)[0]
        else:
            return (Orange.data.Value(self.domain.class_var, int(self.classifier.predict(input)[0])),
                    self.classifier.predict_proba(input)[0])

if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    data = Orange.data.Table('wdbc')
    l1 = GBCLearner(n_estimators=10)
    res = Orange.evaluation.testing.cross_validation([l1],data, 3)
   
    scores = Orange.evaluation.scoring.CA(res)

    for i in range(len(scores)):
        print res.classifier_names[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
