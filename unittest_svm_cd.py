# load a test dataset and test svm
import sklearn.datasets as skd
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from svm_cd import SVMCD
import numpy as np
from sklearn.svm import SVC
import pdb

def main():
    (X,y) = skd.make_classification()
    N = X.shape[0]
    X = np.append(X,np.ones((N,1)),axis=1)
    y = 2*y-1
        
    skf = StratifiedKFold(y,5)
    for train,test in skf:
        X_train = X[train,:]
        y_train = y[train]
        
        X_test = X[test,:]
        y_test = y[test]
        
        C = 0.01
        
        # dual co-ordinate descent SVM
        clf = SVMCD(C)
        clf.fit(X_train,y_train,w_prior=np.ones(21))
        pred = clf.decision_function(X_test)
        score = clf.score(X_test,y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        print score, metrics.auc(fpr, tpr), "//",
        w1  = clf.w;
        
        # standard svm
        clf = SVC(C=C,kernel='linear')
        clf.fit(X_train, y_train) 
        pred = clf.decision_function(X_test)
        score = clf.score(X_test,y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        print score, metrics.auc(fpr, tpr)
        w2 = clf.coef_
        w2.shape = (21,)
        
        
main()        
    
    
    
    
    