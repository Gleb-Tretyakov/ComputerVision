from numpy import ones
from sklearn import svm

def fit_and_classify(train_featues, train_labels, test_features):
	clf = svm.LinearSVC()
	clf.fit(train_featues, train_labels)
	return clf.predict(test_features)
