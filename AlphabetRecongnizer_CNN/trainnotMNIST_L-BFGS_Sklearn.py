from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time
from matplotlib import pyplot as plt

try:
    with open('notMNIST.pickle','rb') as f:
        data_map = pickle.load(f)

        train_dataset = data_map['train_dataset']
        train_label = data_map['train_label']
        print("current shapes : ", train_dataset.shape, train_label.shape)
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
    print("unable to process data map pickle : ", e)

start_time = time()

logreg = LogisticRegression(solver = 'lbfgs',multi_class='multinomial', max_iter=200)
logreg.fit(train_dataset,train_label)
'''
 # gives the set of parameters theta (784)  the model has learned to classify A against the rest
 and of course there are 10 such 784 len parameters each tuned to each letter
 the index 0 gives the coeff for the first "1vs rest" classfier that we build
'''

print("Paramterts theta learned are : ", len(logreg.coef_[0]))
print("training set score : ", logreg.score(train_dataset,train_label)) # if this score is too high that could mean you are overfitting
print("cross validation score : ", logreg.score(cv_dataset,cv_label))

''' working on the test set to evaluate performance
so Y_pred  o/p = class 4 ie e for xi, yi particular test example
whereas Y_prob gives  o/p = [***,***,*** ***,***,***,***,*** ,***,***] ie e for xi, yi particular test example
ie a total of (6000,10) shape
'''
Y_pred = logreg.predict(test_dataset)  # give the actual values label predicted 0.....9

Y_prob = logreg.predict_proba(test_dataset)    # gives the prob for each class for a given test set example
print(Y_prob[0:3])

Y_confidence = []
for a in Y_prob:
    Y_confidence.append(max(a))

print('test accuracy score [out of 6000] :',
      metrics.accuracy_score(test_label,Y_pred, normalize=False), '---- %',
      metrics.accuracy_score(test_label,Y_pred)*100)

end_time = time()

print("time taken :" , end_time-start_time)

'''
NOTE : as stated Y_prob gives 6000 lists each of size 10 (10 being the class size)
        here each list has probabilites/confidence as to what the output alphabet is

NOTE : and Y_pred automatically output the max(currentList) to show that it predicted that as the output

NOTE : since we are concerned with only picking the max from the list and then NOT checking the confidence
        value , threshold doesn't play any role in this case.

        basically its useless seeing the ROC curve
'''

fpr, tpr, thresholds = metrics.roc_curve(Y_pred, Y_confidence,pos_label=2)
plt.title('Receiver Operating Characteristic %s'%metrics.auc(fpr, tpr))
plt.plot(fpr, tpr, 'b')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()









