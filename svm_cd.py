# Linear SVM using dual co-ordinate descent method
# API based on sklearn.svm.SVC [1]
# in addition to the standard C parameter, the learner also takes
# a prior set of weights  w_prior
# based on [2],[3]
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# [2] Hsieh, Cho-Jui, et al. "A dual coordinate descent method for large-scale linear SVM." Proceedings of the 25th international conference on Machine learning. ACM, 2008.
# [3] Gopal, Siddharth, and Yiming Yang. "Recursive regularization for large-scale classification with hierarchical and graphical dependencies." Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2013.

import numpy as np
from numpy.linalg import norm
import scipy.sparse
from random import shuffle
from sklearn.datasets import load_svmlight_file
import sys
import pdb
import time

# NOTE:
# in fit() precomputing XY = X*y which occupies additional memory. Maybe counterproductive in large scale problems. 
class SVMCD:
    
    def __init__(self,C):        
        self.C = C  # risk vs regularization tradeoff of standard SVM
        self.max_iter = 200 # maximum # of outer iteration steps
        self.tol = 1e-15 # required tolerance of optimization solution
    

    def fit(self,X,y,w_prior=None):
        """
        Train Linear SVM using dual co-ordinate descent.
        
        Input: 
        X -- Feature/Design matrix, numpy.array of size (N,D) 
        y -- Label vector, numpy.array of size (N,1)
        w_prior: The prior weights vector numpy.array of size (D,)
            D = Dimension of the input feature space
            N = Number of training examples
            
            
        Processing: 
        Learns the weight vector and dual variables of the linear SVM and saves
        them in the class variables w & alpha respectively
        w -- numpy.array of size (D)
        alpha -- numpy.array of size (N)  
        """
        U = self.C
        (N,D) = X.shape
        
        # initialize W
        if w_prior == None:
            w_prior = np.zeros(D)
        w = w_prior
        w.shape = (1,D)
        alpha = np.zeros((1,N))
        
        #precomputes Q_ii
        Q = np.zeros(N)
        xy_list = []
        for i in range(0,N):
            Q[i] = (X[i,:].dot(X[i,:].T))[0,0]
            xy_list.append(X[i,:]*y[i])
        
        
        for k in range(self.max_iter):
            s = time.time()
            
            alpha_prev = alpha.copy()
            
            # random permutations for efficiency
            indices = range(N) 
            shuffle(indices)
            for i in indices:
                G = y[i]*X[i,:].dot(w.transpose())[0,0]-1
                alpha_old = alpha[0,i]
                alpha[0,i] = min(max(alpha_old-G/Q[i],0),U)
                w = w + (alpha[0,i]-alpha_old)*X[i,:]*y[i]
                
            if  np.linalg.norm(alpha-alpha_prev) < self.tol:
                break
            
            e = time.time()
            self.w = w  # primal weights
            self.alpha = alpha # dual variables
            print "Iteration", k, "Time = ",(e-s), "Obj=", self._objective(X,y,self.w), "Score=", self.score(X,y)
            
        print self.w
        
    def _objective(self,X,y,w):
        """Return the SVM objective function value"""
        nn = norm(w)**2
        fX = np.array(X*(w.T)).ravel()
        loss = sum(np.maximum(0,1-fX*y))
        obj = nn/2 + self.C*loss
        return obj
        
    
    def decision_function(self,X):
        """Evaluate and return the distance of examples from the decision boundary.""" 
        return np.array(X*(self.w.T)).ravel()
    
    def predict(self,X):
        """Evaluate and return the class labels \in {+1,-1}"""
        score = self.decision_function(X)
        pred = np.sign(score)
        return pred
        
    def score(self,X,y):
        """Evaluate the mean accuracy of the classifier for test data (X,y)"""
        pred = self.predict(X)
        return np.mean(pred==y)
    
def main():
    if __name__ == '__main__':
        C = float(sys.argv[1])
        path = sys.argv[2]
        posLbl = int(sys.argv[3])

        X,y = load_svmlight_file(path)
        #pdb.set_trace()
        y = (y==posLbl).astype(int)*2-1
        clf = SVMCD(C)
        clf.fit(X, y)
        acc = clf.score(X, y)
        print "Classification score = ",acc
        print "Training Completed!"
        
main()
