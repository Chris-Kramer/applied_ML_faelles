from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.preprocessing import StandardScaler
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def mover():
    x = input("Continue y/n?: ")
    if x == 'n':
        cont = False
        return cont
    if x == 'y':
        cont = True
        return cont
    else:
        print('Did not register..')
        mover()




def convert_sample(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,[32,32]).numpy()
    image = image.reshape(1,-1)
    return image



data_dir = r'C:\Users\andly\OneDrive\Dokumenter\applied_ML_faelles'
print('Current data dir '+data_dir)

cont = mover()
if cont:


    ds1,ds2,ds3 = tfds.load('patch_camelyon',
                        split=['train[:5%]','test[:2%]','validation[:2%]'],
                        data_dir = data_dir,
                        download=False,
                        batch_size=-1, # All data...no batches needed 
                        as_supervised=True, # So that we easily can transform data to numpy format
                        shuffle_files=True)
    print('Done')

else:
    None


train_dataset       = tfds.as_numpy(ds1)
train_dataset_image = np.vstack(list(map(convert_sample,train_dataset[0])))
train_dataset_image_Scaled = StandardScaler(with_mean=0, with_std=1).fit_transform(train_dataset_image)
train_dataset_label = train_dataset[1].reshape(-1,)    
print(f'Shape of training data features (observations,features): {train_dataset_image_Scaled.shape}')
print(f'Shape of training data labels (observations,): {train_dataset_label.shape}')

validation_dataset  = tfds.as_numpy(ds3)
validation_dataset_image = np.vstack(list(map(convert_sample,validation_dataset[0])))
validation_dataset_image_Scaled = StandardScaler(with_mean=0, with_std=1).fit_transform(validation_dataset_image)
validation_dataset_label = validation_dataset[1].reshape(-1,) 
   
test_dataset        = tfds.as_numpy(ds2)
test_dataset_image = np.vstack(list(map(convert_sample,test_dataset[0])))
test_dataset_image_Scaled = StandardScaler(with_mean=0, with_std=1).fit_transform(test_dataset_image)
test_dataset_label = test_dataset[1].reshape(-1,)


cont = mover()

if cont:
    clf = svm.SVC(kernel='rbf')
    clf.fit(train_dataset_image_Scaled, train_dataset_label)
    y_test_hat = clf.predict(test_dataset_image)

    # Obtain accuracy by using the `accuracy_score` function
    accuracy_linear = accuracy_score(y_test_hat, test_dataset_label )
    # Print results
    print(f'SVM achieved {round(accuracy_linear * 100, 1)}% accuracy.')

else:
    None