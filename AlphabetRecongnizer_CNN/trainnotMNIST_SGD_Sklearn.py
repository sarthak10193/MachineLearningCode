from six.moves import cPickle as pickle
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from time import time
import numpy as np

try:
    with open('notMNIST.pickle','rb') as f:
        data_map = pickle.load(f)

        train_dataset = data_map['train_dataset']
        train_label = data_map['train_label']
        train_dataset = np.reshape(train_dataset, (len(train_dataset),784)) # (16060, 28, 28) -> (16060, 784)
        print(" X TRAIN shape = ", train_dataset.shape)

        cv_dataset = data_map['cv_dataset']
        cv_label = data_map['cv_label']
        cv_dataset = np.reshape(cv_dataset, (len(cv_dataset),784)) # (16060, 28, 28) -> (16060, 784)
        print(" X CV shape = ", cv_dataset.shape)

        test_dataset = data_map['test_dataset']
        test_label = data_map['test_label']
        test_dataset = np.reshape(test_dataset, (len(test_dataset),784)) # (16060, 28, 28) -> (16060, 784)
        print(" X Test shape = ", test_dataset.shape)
except Exception as e:
    print("unbable to process pickle ... error :", e)


start_time = time()

'''
SGD is much faster than l-bfgs and is better for scaling to large machine learning problems
less computationally expensive since we don't sum over the training set every time , we just consider
one training example at a time inside the loop for convergence.

but SGD may not always achieve the exact or as good as a global min as batch GD or l-bfgs
has a lot of hyperparamters which are hard to tune correctly

'''
logreg_SGD = SGDClassifier(n_iter=30, shuffle=True , loss='log' , alpha=0.001)
logreg_SGD.fit(train_dataset,train_label)
#print("Training set score : ",)
#print("cross validation score : " , )

Y_problist = logreg_SGD.predict_proba(test_dataset)

Y_pred = []
for a in Y_problist:
    Y_pred.append(np.argmax(a))

print("test set accuracy [o/f 6000 ]:", metrics.accuracy_score(test_label,Y_pred,normalize=True),
      ' --- no of examples correct  : ', metrics.accuracy_score(test_label,Y_pred, normalize=False))

end_time = time()

print("time taken : ",  end_time-start_time)
