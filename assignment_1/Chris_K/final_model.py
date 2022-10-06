#Import
# Data wrangling
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# Model
from catboost import CatBoostClassifier
# performance measure
from sklearn.metrics import accuracy_score
# Grid search
from sklearn.model_selection import GridSearchCV

# ------ Load data ------
# test data (no labels (y_test) since we that way can't see the results)
X_test = np.load("../Common/data/X_test.npy")
# Validation and training
X_train, X_val, y_train, y_val = train_test_split(
                                                  np.load("../Common/data/X_train.npy"), # X data
                                                  np.load("../Common/data/y_train.npy"), # y data (labels)
                                                  test_size = 0.5,
                                                  random_state = 42
                                                  )
# ------ Create and train model ------ 
final_model = CatBoostClassifier(
                           verbose = 100, # Print training process every 100 iteration
                           random_state = 42, # For reproducibility
                           early_stopping_rounds = 10 #, No default - Preventing overfitting
                           )
best_params = {

}
final_model = final_model.set_params(best_params)
final_model.fit(np.concatenate(X_train, X_val), np.concatenate(y_train, y_val))

# ------ Create and save predictions ------
final_model_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_train, final_model_pred)



y_test_hat = final_model.predict(X_test)
y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(y_test_hat))),
    'Category': y_test_hat,
})


# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Category'])

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('y_test_hat.csv', index=False)