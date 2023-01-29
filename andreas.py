from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
from sklearn import tree, ensemble
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.preprocessing import StandardScaler
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import time

def convert_sample(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,[32,32]).numpy()
    image = image.reshape(1,-1)
    return image


data_dir = r'C:\Users\andly'
print('Current data dir '+data_dir)


ds1,ds2,ds3 = tfds.load('patch_camelyon',
                    split=['train[:2%]','test[:2%]','validation[:2%]'],
                    data_dir = data_dir,
                    download=False,
                    batch_size=-1, # All data...no batches needed 
                    as_supervised=True, # So that we easily can transform data to numpy format
                    shuffle_files=True)
print('Done Loading Data')


train_dataset = tfds.as_numpy(ds1) # FULL DATA
train_dataset_image = np.vstack(list(map(convert_sample,train_dataset[0]))) # <-- This is the X
train_dataset_image_Scaled = StandardScaler(with_mean=0, with_std=1).fit_transform(train_dataset_image)
train_dataset_label = train_dataset[1].reshape(-1,) # <-- This is y   
print(f'Shape of training data features (observations,features): {train_dataset_image_Scaled.shape}')
print(f'Shape of training data labels (observations,): {train_dataset_label.shape}')

validation_dataset = tfds.as_numpy(ds3)
validation_dataset_image = np.vstack(list(map(convert_sample,validation_dataset[0])))
validation_dataset_image_Scaled = StandardScaler(with_mean=0, with_std=1).fit_transform(validation_dataset_image)
validation_dataset_label = validation_dataset[1].reshape(-1,) 

test_dataset = tfds.as_numpy(ds2)
test_dataset_image = np.vstack(list(map(convert_sample,test_dataset[0]))) # <-- X_test
test_dataset_image_Scaled = StandardScaler(with_mean=0, with_std=1).fit_transform(test_dataset_image)
test_dataset_label = test_dataset[1].reshape(-1,)
print("Done spliting data")


from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean, std

xgb = XGBClassifier(tree_method="gpu_hist", objective='binary:logistic', nthread=4, seed=42)


# Make a dict. of hyperparamter values to search


search_space = { # Keys equals to names of hyperparameters 
"n_estimators": list(range(60,400,60)),
"max_depth": [2,10,1],
'learning_rate': [0.1, 0.01, 0.05]
}
# Number of candidates: 54


# Make a GridSearchCV object
GS = GridSearchCV(estimator = xgb,
                    param_grid = search_space,
                    scoring = 'roc_auc', #sklearn.metrics.SCORERS.keys()
                    n_jobs= 10,
                    cv = 10,
                    verbose=True
                    )
    

# Fitting the model
st = time.time()

GS.fit(train_dataset_image_Scaled, train_dataset_label)

#model2.fit(train_dataset_image_Scaled, train_dataset_label, eval_metric='logloss', eval_set=evalset)

et = time.time()
elapsed_time = et - st

print("Done Fitting. Time taken: ",elapsed_time)

print("Best estimator: ", GS.best_estimator_)

print("Best params.: ", GS.best_params_)

print("Best score: ", GS.best_score_)





  