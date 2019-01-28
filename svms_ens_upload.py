"""
SVMs and Ensemble Learning on the Default data set
"""

# Here we import all we will need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier




path_to_carseats = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Carseats.csv'



"""
We use the ifelse() function to create a variable, called High, which takes
on a value of Yes if the Sales variable exceeds 8, and takes on a value of No
otherwise. We’ll append this onto our dataFrame using the .map() function:
"""



# Let us define X and y



# We first split the observations into a training set and a test set:



# Let us train a linear SVC


# - (11+13)/200 = 0.88



# We can also get the class probabilities:



##########################################################################
############ Here we construct the ROC curve for the tree ################
##########################################################################
y_score = linsvc.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='orange',
         label='ROC curve (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for our Linear SVC')
plt.legend(loc="lower right")
##########################################################################
##########################################################################

# Let us fit a SVC with radial kernel




#svc with grid search




#svc in a bagging classifier



# We can use oob to find the optimal number of estimators


    
# svc, random forest, and logistic regression in a voting classifier


    
# CV in the voting classifier (OBS takes approx. 20 min)
    
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(random_state=181)
svm_clf = SVC(kernel='linear', probability=True, random_state=181)

    
voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft')
    
params = {'lr__C': [0.01, 0.1, 1, 10],
          'svc__C': [0.01, 0.1, 1, 10],
          'rf__max_features': [6, 'sqrt', 'log2']}

grid = GridSearchCV(estimator=voting_clf, param_grid=params, cv=5)

grid.fit(X_train,y_train)

print (grid.best_params_)
# {'lr__C': 0.01, 'rf__max_features': 'sqrt', 'svc__C': 1}

pred = grid.predict(X_test)

accuracy_score(y_test, pred)
# 0.835
