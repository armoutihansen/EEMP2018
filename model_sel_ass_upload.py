"""
Model selection and assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold   
from sklearn.model_selection import RepeatedKFold



# Let us generate the data:














                                                    



# Let's start with regression














    
# Let us plot the train and test errors:
    













# Let's see how close we get to select the best model with CV:

## Before, let us see how CV is done with sklearn






## Compare our score to Err_Tr






# Now let's do CV for model selection:











# Let us try to plug these into our plot







# Now, let's try LOOCV and 10-fold CV:









# Let us try to plug these into our plot












# Let us try to plug these into our plot














# Let us try to plug these into our plot









### Now we'll work with the kNN classifier ###
""" Try on your own, but notice the following changes compared to regression:
# 1. We cannot and should no longer use MSE
    Insted, directly after using the fit method, use train_loss = -1*clf.score(X_train, y_train).
    Thus, you do not need to specify y_hat anymore.
    Here: clf refers to classifier, and should be changed if you name it something else
    We use -1 times the score, as we want the MIS-classification rate.
    Remember also to calculate the test misclassification rate
# 2. When you do CV, you do no longer need to specify a scoring parameter
    The classification rate is returned, and you can get the MIS-classification rate 
    by multiplying with -1
"""


