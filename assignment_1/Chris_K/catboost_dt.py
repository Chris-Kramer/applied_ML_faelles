#!/usr/bin/env python3

"""
The line above is a shebang and is used for reproducibility, so this script can be run on Unix systems

Catboost use different names for parameters compared to sklearns model
here is a nice guide for catboost https://www.kaggle.com/code/mitribunskiy/tutorial-catboost-overview/notebook
Otherwise the official homepage have more information https://catboost.ai/en/docs/ 
"""
# ------ Import -------
# Data wrangling
import numpy as np
from sklearn.model_selection import train_test_split
# Model
from catboost import CatBoostClassifier
# performance measure
from sklearn.metrics import accuracy_score


# ------ Load data ------
# test data (no labels (y_test) since we that way can't see the results)
X_test = np.load("../Common/data/X_test.npy")
# Validation and training
X_train, X_val, y_train, y_val = train_test_split(
                                                  np.load("../Common/data/X_train.npy"), # X data
                                                  np.load("../Common/data/y_train.npy"), # y data (labels)
                                                  test_size = 0.2,
                                                  random_state = 42
                                                  )

# ------ Create and test model ------
model = CatBoostClassifier(
                           verbose = 100, # Print training process every 100 iteration
                           random_state = 42 # For reproducibility
                           )

model.fit(X_val, y_val, plot = True)
y_train_hat = model.predict(X_train)
accuracy = accuracy_score(y_train, y_train_hat)
print(accuracy * 100)

# ----- Train final model ------

