import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree






#*******************************
#B1
all_data = pd.read_csv('mushrooms.csv')
all_labels = all_data['class']
all_data.pop('class')

n_overall = len(all_data)

X_train = all_data[0:int(0.7 * n_overall)]
Y_train = all_labels[0:int(0.7 * n_overall)]

X_verification = all_data[int(0.7 * n_overall):int(0.9 * n_overall)]
Y_verification = all_labels[int(0.7 * n_overall) : int(0.9 * n_overall)]

X_test = all_data[int(0.9 * n_overall):]
Y_test= all_labels[ int(0.9 * n_overall):]
maximum_depths = [4,6,8,10,12,14,16,18,20]
losses = []
models = []
y_vectors = Y_verification.to_numpy()
for depth in maximum_depths:
    model = tree.DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, Y_train)
    models.append(model)
    error_count = 0
    y_predict = model.predict(X_verification)
    for i in range(len(X_verification)):
        if y_predict[i] != y_vectors[i]:
            error_count += 1
    losses.append(error_count/len(X_verification))
plt.figure(3)
plt.plot(maximum_depths,losses,color = 'b')
plt.suptitle('Loss decision tree based on max depth')
plt.show()
#the figure will be saved in the figures folder

best_model = models[losses.index(min(losses))]
error_count = 0
y_test_vectors = Y_test.to_numpy()
y_predict = best_model.predict(X_test)
for i in range(len(X_test)):
    if y_predict[i] != y_test_vectors[i]:
        error_count += 1

print("the loss of the best model on test set is "+ str(error_count/len(X_test)))
#the loss of the best model on test set is 0.025830258302583026

"""
as seen in the figure, the lowest errors occur at max-depth 6 and 20. while 20 being large which makes our tree 
complicated, 6 is a decent depth for our decision tree.
I also tried the algorithm above with the models having zero random state, which indicated that after
max depth = 6 all depths had the same value, which indicates that max-depth can be considered as optimal.
"""



