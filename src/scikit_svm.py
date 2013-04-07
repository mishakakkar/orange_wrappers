import Orange
import numpy as np
from sklearn import svm

class ScikitSVMLearner(Orange.classification.Learner):
    """
    Wrapper for SVM from scikit-learn.
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

        if len(np.unique(Y)) == 1:
            return Orange.classification.majority.MajorityLearner(data)

        svc = svm.SVC(**self.kwargs)
        svc.fit(X, Y)

        return ScikitSVMClassifier(classifier=svc, domain = data.domain)

class ScikitSVMClassifier(Orange.classification.Classifier):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, example, result_type=Orange.core.GetValue):
        input = np.array([float(example[i]) for i in xrange(len(example)-1)])

        if result_type == Orange.core.GetValue:
            return Orange.data.Value(self.domain.class_var, int(self.classifier.predict(input)[0]))
        elif result_type == Orange.core.GetProbabilities:
            return self.classifier.predict_proba(input)[0]
        else:
            return (Orange.data.Value(self.domain.class_var, int(self.classifier.predict(input)[0])), self.classifier.predict_proba(input)[0])

if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    data = Orange.data.Table('glass')
    l1 = ScikitSVMLearner(kernel='linear', C=1, probability=True)
    res = Orange.evaluation.testing.cross_validation([l1],data, 3)

    scores = Orange.evaluation.scoring.CA(res)

    for i in range(len(scores)):
        print res.classifier_names[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
