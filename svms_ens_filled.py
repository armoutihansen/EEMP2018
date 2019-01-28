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
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, auc, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier




path_to_carseats = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Carseats.csv'

df = pd.read_csv(path_to_carseats, index_col=0)

print(df.describe())

print(df.info())

pd.scatter_matrix(df, alpha=0.2)

"""
We use the ifelse() function to create a variable, called High, which takes
on a value of Yes if the Sales variable exceeds 8, and takes on a value of No
otherwise. We’ll append this onto our dataFrame using the .map() function:
"""

df['High'] = df['Sales'].map(lambda x: 1 if x>8 else 0)
df = pd.get_dummies(df, drop_first=True)

print(df.info())

# Let us define X and y

X = df.drop(['High', 'Sales'], axis=1).values
y = df['High'].values

# We first split the observations into a training set and a test set:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=181)

# Let us train a linear SVC

linsvc = SVC(kernel='linear', random_state=181)

linsvc.fit(X_train, y_train)

tr_pred = linsvc.predict(X_train)

accuracy_score(y_train, tr_pred)

pred = linsvc.predict(X_test)

accuracy_score(y_test, pred)

cm = pd.DataFrame(confusion_matrix(y_test, pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
print(cm)
#1 - (11+13)/200 = 0.88

linsvc.score(X_test, y_test)

# We can also get the class probabilities:

linsvc = SVC(kernel='linear', probability=True, random_state=181)

linsvc.fit(X_train, y_train)

linsvc.predict_proba(X_test.head(1))

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

r_svc = SVC(kernel='rbf', gamma=0.001, random_state=181)

r_svc.fit(X_train, y_train)

tr_pred = r_svc.predict(X_train)

accuracy_score(y_train, tr_pred)

pred = r_svc.predict(X_test)

accuracy_score(y_test, pred)


#svc with grid search

svc = SVC(random_state=181)

parameters = {'kernel':('linear', 'rbf'), 'C':(0.001, 0.01,0.1,1,10),'gamma': (0.0001,0.001,0.01,1,10)}

clf = GridSearchCV(svc, parameters, cv=5)

clf.fit(X_train,y_train)

clf.best_estimator_

clf.best_score_

print("accuracy:"+str(np.average(cross_val_score(clf, X_train, y_train, scoring='accuracy'))))


#svc in a bagging classifier

bag = BaggingClassifier(SVC(kernel='linear'), n_estimators=10, random_state=181)

bag.fit(X_train, y_train)

pred = bag.predict(X_test)

accuracy_score(y_test, pred)

# We can use oob to find the optimal number of estimators

best_score= 0
best_model = 0
for i in np.arange(10,20):
    bag = BaggingClassifier(SVC(kernel='linear'), n_estimators=i, oob_score=True, random_state=181)
    bag.fit(X_train, y_train)
    acc_score = bag.oob_score_
    if acc_score > best_score:
        best_score = acc_score
        best_model = bag
        
best_model
pred = best_model.predict(X_test)
accuracy_score(y_test, pred)
    
# svc, random forest, and logistic regression in a voting classifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(random_state=181)
svm_clf = SVC(kernel='linear', random_state=181)

voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, pred))
    
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
