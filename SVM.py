import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC


##A1
## your code to load data
X = pd.read_csv('data_banknote_authentication.csv')

# X = ...
# Y = ...
Y = X['label']
X.pop('label')
# adding the column with one's as discussed in the class
X['b'] = pd.Series([1 for i in range(len(X.index))], index=X.index)


# Your code to split the data (with no randomization)
# X_train = ...
# Y_train = ...


# X_test = ...
# Y_test = ..
border = int(0.8 * len(X))
X_train = X[0:border]
Y_train = Y[0:border]
X_test = X[border:]
Y_test = Y[border:]




#A2

## Implement the algorithm here

def perceptron(X_tr, Y_tr, X_te, Y_te, max_iter=50000):
    feature_vectors = X_tr.to_numpy()
    labels = Y_tr.to_numpy()
    test_vectors = X_te.to_numpy()
    test_labels = Y_te.to_numpy()
    loss_history = list()
    w = np.zeros(len(X_tr.columns),dtype = float)
    def find_incostintency(w,feature_vectors, labels):
        #finding a training data which violates the separability constraint
        starting_point = random.randrange(0,len(feature_vectors))
        n = len(feature_vectors)
        for i in range(n):
            index = (i + starting_point) % n
            if labels[index] * (np.dot(w,feature_vectors[index])) <= 0:
                return index
        return None
    def loss(w,test_vectors,test_labels):
        loss_counts = 0
        for i in range(len(test_vectors)):
            if test_labels[i] * (np.dot(w,test_vectors[i])) < 0:
                loss_counts+= 1
        return loss_counts/len(test_vectors)
    iter = 0
    final_loss = 0
    while iter <= max_iter:
        if iter%500 == 0 and iter !=0:
            loss_history.append(loss(w,test_vectors,test_labels))
        violating_data = find_incostintency(w,feature_vectors,labels)
        if violating_data!= None:
            w = np.add(w,labels[violating_data] * feature_vectors[violating_data])
            iter += 1
        else:
            break
    final_loss = loss(w, test_vectors, test_labels)
    return w,loss_history,final_loss
w,loss_history,final_loss = perceptron(X_train,Y_train,X_test,Y_test)
#
print("the output weight vector of the perceptron algorithm  is: ")
print(w)
print("the final loss of the perceptron algortihm is : ",final_loss)
#  the output weight vector of the perceptron algorithm  is:
#  [-293.7849    -158.695979  -186.2657656  -12.9038152  261.       ]
#  the final loss of the perceptron algortihm is :  0.014545454545454545
print("plotting the errors of the perceptron algorithm:")
plt.figure(1)
x_axis = [i * 500 for i in range(1,len(loss_history) + 1)]
plt.plot(x_axis,loss_history,color = 'r')
plt.suptitle('Loss of perceptron for linear classification')
plt.show()
# the figure will be saved in the figures folder


#*******************************
#A3
mapped_trained_data = X_train.assign(new_feaature = lambda x: np.power(X_train['feature 4'],3))
mapped_test_data = X_test.assign(new_feaature = lambda x: np.power(X_test['feature 4'],3))
w,loss_history,final_loss = perceptron(mapped_trained_data,Y_train,mapped_test_data,Y_test)

print("the output weight vector of the perceptron algorithm for non-linear classification is: ")
print(w)
print("the final loss of the perceptron algortihm for non-linear classification is : ",final_loss)
#
#  the output weight vector of the perceptron algorithm for non-linear classification is:
#  [-1.18371975e+04 -6.19703641e+03 -7.95634267e+03 -1.47238805e+03
#    9.48300000e+03 -1.54897160e+00]
#  the final loss of the perceptron algortihm for non-linear classification is :  0.02181818181818182
print("plotting the errors of the perceptron for non-linear classification algorithm:")
plt.figure(2)
x_axis = [i * 500 for i in range(1,len(loss_history) + 1)]
plt.plot(x_axis,loss_history,color = 'r')
plt.suptitle('Loss of perceptron for linear classification')
plt.show()
# the figure will be saved in the figures folder



#*******************************
#A4

# Train a SVM model, report final errors

emp_loss, true_loss = 0, 0
weights = 0
X = pd.read_csv('data_banknote_authentication.csv')
Y = X['label']
X.pop('label')
border = int(0.8 * len(X))
X_train = X[0:border]
Y_train = Y[0:border]
X_test = X[border:]
Y_test = Y[border:]

svmclassifier = SVC(kernel='linear',max_iter=50000)
svmclassifier.fit(X_train, Y_train)
print("SVM linear classifier coefficients:")
print(svmclassifier.coef_)
# # [[-2.55893415 -1.49168504 -1.81294106 -0.24726297]]
#
# #returning the error
train_vectors = X_train.to_numpy()
train_labels = Y_train.to_numpy()
test_vectors = X_test.to_numpy()
test_labels = Y_test.to_numpy()
#
y_pred = svmclassifier.predict(X_train)
error_counts = 0
for i in range(len(train_vectors)):
    if train_labels[i] != y_pred[i]:
        error_counts += 1

print("empirical error for SVM linear classification:")
print(error_counts/len(train_vectors))
#0.009115770282588878
#
y_pred = svmclassifier.predict(X_test)
error_counts = 0
for i in range(len(test_vectors)):
    if test_labels[i] != y_pred[i]:
        error_counts += 1

print("error for SVM linear classification on tests:")
print(error_counts/len(test_vectors))
#0.014545454545454545



#*******************************
#A5

svmclassifier2 = SVC(kernel='linear',max_iter=50000)
mapped_trained_data = X_train.assign(new_feaature = lambda x: np.power(X_train['feature 4'],3))
mapped_test_data = X_test.assign(new_feaature = lambda x: np.power(X_test['feature 4'],3))


svmclassifier2.fit(mapped_trained_data, Y_train)
print("SVM general classifier coefficients:")
print(svmclassifier2.coef_)
# [[-2.45950847e+00 -1.41875226e+00 -1.72020474e+00 -1.92110426e-01 -6.69808618e-04]]

#returning the error
train_vectors = mapped_trained_data.to_numpy()
train_labels = Y_train.to_numpy()
test_vectors = mapped_test_data.to_numpy()
test_labels = Y_test.to_numpy()

y_pred = svmclassifier2.predict(mapped_trained_data)
error_counts = 0
for i in range(len(train_vectors)):
    if train_labels[i] != y_pred[i]:
        error_counts += 1

print("empirical error for SVM general classification:")
print(error_counts/len(train_vectors))
#0.010938924339106655

y_pred = svmclassifier2.predict(mapped_test_data)
error_counts = 0
for i in range(len(test_vectors)):
    if test_labels[i] != y_pred[i]:
        error_counts += 1

print("error for SVM general classification on tests:")
print(error_counts/len(test_vectors))
#0.014545454545454545

#*******************************
#A6

"""
conclusion:
as we witnessed,the weights of the perceptron algorithm are greater in value than svm, implying that the svm algorithm probably 
normalizes the elements that are utilized in it.
as predicted, the SVM errors(loss) had a lower value compared to perceptron errors. adding the extra feature caused the
perceptron error to decline, but the SVM error remained unchanged.
Also the empirical risk increased after adding the extra feature.
"""
