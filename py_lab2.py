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



# Let us see the summary statistics



# Looks like we need to create dummies for 'default' and 'student'














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
    
















# Let us perform logistic regression with statmodels











# Let us perform the regressions with scikit-learn















# Let us try a different algorithm







# Let us now perform LDA on the data


# We can check the possible classes:

# We can check the priors, i.e. our estimates of Pr(Y=k):

# We can check our estimates of mu_k:



# Let us reproduce the confusion matrix



                                    






# We can also print a classification report:


# Note: Precision is the rate of prediction that corresponds to the truth
## Here: we predict 9645+254=9899 no defaults from which 9645 were no defaults
## Here: we predict 22+79=101 defaults from which 79 where defaults 
# Note: Recall is one minus the false positive and false negative rate, respectively 

# Let us try a different threshold than 50%, e.g. 20%




    
    
    
    







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






                   




                                    



# Now, let us split the data into a training and test data set


                                                    
















