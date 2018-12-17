"""
Linear classification on the Default data set
"""
# Here we import all we will need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

path_to_data = 'https://raw.githubusercontent.com/jeshan49/eemp2/master/Default.csv'

# Let us import the default data

default = pd.read_csv(path_to_data, index_col=0)

# Let us see the summary statistics

default.describe()

# Looks like we need to create dummies for 'default' and 'student'

default['d_default'] = pd.get_dummies(default['default'], drop_first=True)

default['d_student'] = pd.get_dummies(default['student'], drop_first=True)

default['d_student2'] = pd.get_dummies(default['student']).loc[:,'Yes']

default['d_student3'] = default['student'].apply(lambda val: 1 if val == 'Yes' else 0)

default.head()

default.describe()


##########################################################################
##### Here we construct the scatter and box plots from the lecture #######
##########################################################################
fig, axs = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[2, 1, 1]})

# Take a fraction of the samples where target value (default) is 'no'
df_no = default[default['d_default'] == 0].sample(frac=0.15)
# Take all samples  where target value is 'yes'
df_yes = default[default['d_default'] == 1]
# Join the samples in one dataframe
df_ = df_no.append(df_yes)

# Let us construct the scatter plot
axs[0].scatter(df_[df_.default == 'Yes'].balance, df_[df_.default == 'Yes'].income,
   s=40, c='orange', marker='+', linewidths=1, label='Default = Yes')
axs[0].scatter(df_[df_.default == 'No'].balance, df_[df_.default == 'No'].income,
   s=40, marker='o', linewidths='1', edgecolors='lightblue', facecolors='white',
   alpha=.6, label = 'Default = No')

axs[0].set_ylim(ymin=0)
axs[0].set_ylabel('Income')
axs[0].set_xlim(xmin=-100)
axs[0].set_xlabel('Balance')
axs[0].legend()

# Let us now construct the two boxplots
c_palette = {'No':'lightblue', 'Yes':'orange'}
sns.boxplot('default', 'balance', data=default, orient='v', ax=axs[1],
            palette=c_palette)
sns.boxplot('default', 'income', data=default, orient='v', ax=axs[2],
            palette=c_palette)
##########################################################################
##########################################################################

##########################################################################
####### Here we construct the regression plots from the lecture ##########
##########################################################################
fig, axs = plt.subplots(2, 1)
sns.regplot('balance', 'd_default', data=default,
            scatter_kws={'color':'orange'}, line_kws={'color':'blue'}, ax=axs[0])

sns.regplot('balance', 'd_default', data=default, ci=None, logistic=True,
            scatter_kws={'color':'orange'}, line_kws={'color':'blue'}, ax=axs[1])

axs[0].set_title('Linear Regression')
axs[1].set_title('Logistic Regression')

for ax in axs:
    ax.set_ylabel('Prob of Default')
    ax.set_xlabel('Balance')
##########################################################################
##########################################################################

# Let us perform linear regression with statsmodels
    
y = default['d_default']

X = default[['balance', 'income', 'd_student']]

linreg = sm.OLS(y, X).fit()

linreg.summary()

print(linreg.summary().tables[1])

X = sm.add_constant(X)

linreg2 = sm.OLS(y, X).fit()

linreg2.summary()

# Let us perform logistic regression with statmodels

logreg = sm.Logit(y, X[['const', 'balance']]).fit()

logreg.summary()

print(logreg.summary().tables[1])

logreg2 = sm.Logit(y, X).fit()

logreg2.summary()

# Let us perform the regressions with scikit-learn

X = default[['balance', 'income', 'd_student']]

sk_linreg = LinearRegression().fit(X, y)

print(linreg2.summary().tables[1])
sk_linreg.coef_
sk_linreg.intercept_

sk_logreg = LogisticRegression().fit(X, y)

print(logreg2.summary().tables[1])
sk_logreg.coef_
sk_logreg.intercept_

# Let us try a different algorithm
X = default[['balance', 'income', 'd_student']]
sk_logreg2 = LogisticRegression(solver='newton-cg').fit(X, y)

print(logreg2.summary().tables[1])
sk_logreg2.coef_
sk_logreg2.intercept_

# Let us now perform LDA on the data
lda = LinearDiscriminantAnalysis().fit(X, default['default'])

# We can check the possible classes:
lda.classes_
# We can check the priors, i.e. our estimates of Pr(Y=k):
print(pd.DataFrame(lda.priors_, index=['No', 'Yes'], columns=['Pr(Y)']))
# We can check our estimates of mu_k:
print(pd.DataFrame(lda.means_, index = ['No', 'Yes'],
                   columns = X.columns))

# Let us reproduce the confusion matrix
y_hat = lda.predict(X)

print(pd.DataFrame(confusion_matrix(default['default'], y_hat).T, 
                                    index = ['No', 'Yes'], columns = ['No', 'Yes']))

fal_pos_rate = confusion_matrix(default['default'], y_hat).T[1,0]/np.sum(confusion_matrix(default['default'], y_hat).T[:,0])
print(fal_pos_rate)
fal_neg_rate = confusion_matrix(default['default'], y_hat).T[0,1]/np.sum(confusion_matrix(default['default'], y_hat).T[:,1])
print(fal_neg_rate)

# We can also print a classification report:
print(classification_report(default['default'], y_hat))

# Note: Precision is the rate of prediction that corresponds to the truth
## Here: we predict 9645+254=9899 no defaults from which 9645 were no defaults
## Here: we predict 22+79=101 defaults from which 79 where defaults 
# Note: Recall is one minus the false positive and false negative rate, respectively 

# Let us try a different threshold than 50%, e.g. 20%

prob_hat = lda.predict_proba(X)[:,1]
new_y_hat = []
for i in prob_hat:
    if i>0.2:
        new_y_hat.append('Yes')
    else:
        new_y_hat.append('No')
new_y_hat = np.array(new_y_hat)

print(pd.DataFrame(confusion_matrix(default['default'], new_y_hat).T, 
                                    index = ['No', 'Yes'], columns = ['No', 'Yes']))

print(classification_report(default['default'], new_y_hat))

##########################################################################
########## Here we construct the ROC curve from the lecture ##############
##########################################################################
y_score = lda.fit(X, y).decision_function(X)

fpr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='orange',
         label='ROC curve (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for our LDA')
plt.legend(loc="lower right")
##########################################################################
##########################################################################

# Let us turn to QDA now

qda = QuadraticDiscriminantAnalysis().fit(X, default['default'])

print(pd.DataFrame(qda.priors_, index=['No', 'Yes'], columns=['Pr(Y)']))

print(pd.DataFrame(qda.means_, index = ['No', 'Yes'],
                   columns = X.columns))

y_hat = qda.predict(X)

print(pd.DataFrame(confusion_matrix(default['default'], y_hat).T, 
                                    index = ['No', 'Yes'], columns = ['No', 'Yes']))

print(classification_report(default['default'], y_hat))

# Now, let us split the data into a training and test data set

X_train, X_test, y_train, y_test = train_test_split(X, default['default'],
                                                    test_size=0.3, random_state=181)

logreg = LogisticRegression(solver='newton-cg').fit(X_train, y_train)
logreg_y_hat = logreg.predict(X_test)
precision_score(y_test, logreg_y_hat, labels=['No', 'Yes'], pos_label='No')

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
lda_y_hat = lda.predict(X_test)
precision_score(y_test, lda_y_hat, labels=['No', 'Yes'], pos_label='No')

qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
qda_y_hat = qda.predict(X_test)
precision_score(y_test, qda_y_hat, labels=['No', 'Yes'], pos_label='No')

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
knn_y_hat = knn.predict(X_test)
precision_score(y_test, knn_y_hat, labels=['No', 'Yes'], pos_label='No')
