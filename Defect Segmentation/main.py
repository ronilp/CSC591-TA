import os
import time
import warnings
from random import shuffle

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from data_generator import Surface_Generator
from metrics import dice_coef_loss, dice_coef
from unet import get_unet

warnings.filterwarnings("ignore")


# Not all data in the dataset has defects. We only use the images which have defects
# This function takes the dataset_type as a parameter. You can pass "Train" or "Test"
# as argument to get the appropriate dataset
def load_data(dataset_type="Train"):
    file_list = {}
    defect_map = {}
    file_name = []
    file_mask = []
    count = 0
    num_classes = 6

    data_dir = "data"
    for x in range(1, num_classes + 1):
        path = os.path.join(os.path.join(data_dir, "Class" + str(x)), dataset_type)
        df = pd.read_fwf(path + "/Label/Labels.txt")
        count = 0
        for i in range(0, len(df)):
            curr_file = path + "/" + str(df.iloc[i][2])
            if (df.iloc[i][1] == 1):
                file_list[curr_file] = path + "/Label/" + str(df.iloc[i][4])
                defect_map[curr_file] = 1
            else:
                fnametest = str(df.iloc[i][2]).split(".")
                file_list[curr_file] = str(path + "/Label/" + fnametest[0] + "_label.PNG")
                defect_map[curr_file] = 0

    items = list(file_list.keys())
    shuffle(items)
    for key in items:
        if ((not os.path.exists(key)) or (not os.path.exists(file_list[key]))):
            # print ("Missing mask for ", key)
            continue

        if defect_map[key] == 1:
            file_name.append(key)
            file_mask.append(file_list[key])
        elif count < 80 * num_classes:
            file_name.append(key)
            file_mask.append(file_list[key])
            count = count + 1

    return file_name, file_mask


if __name__ == "__main__":
    # Since we already have a split for training and test set,
    # we just need to split training set to get a validation set

    # Load training data
    X, Y = load_data("Train")

    # Split the original training data to get training and validation set
    # to get X_train, X_val, y_train, y_val
    # YOUR CODE HERE
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20)

    # Convert to numpy arrays
    # YOUR CODE HERE
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    model = get_unet()
    batch_size = 8
    num_epochs = 50
    # Compile the model
    model.compile(loss=dice_coef_loss, optimizer=Adam(lr=0.0055), metrics=[dice_coef])

    # Create generator objects for training and validation
    # YOUR CODE HERE
    # training_batch_generator = Surface_Generator(...)
    # validation_batch_generator = Surface_Generator(...)

    num_training_samples = len(X_train)
    num_validation_samples = len(X_val)
    training_batch_generator = Surface_Generator(X_train, y_train, batch_size)
    validation_batch_generator = Surface_Generator(X_val, y_val, batch_size)

    # callbacks for saving models and early stopping
    checkpointer = ModelCheckpoint("unet_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor=dice_coef,
                                   verbose=1,
                                   save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    # Fit model
    # This will take ~1.5-2 minutes per epoch on a GPU
    stmillis = int(round(time.time() * 1000))
    history = model.fit_generator(generator=training_batch_generator,
                                  steps_per_epoch=(num_training_samples // batch_size),
                                  epochs=num_epochs,
                                  verbose=1,
                                  validation_data=validation_batch_generator,
                                  validation_steps=(num_validation_samples // batch_size),
                                  use_multiprocessing=True,
                                  workers=5,
                                  max_queue_size=1,
                                  callbacks=[checkpointer, early_stopping])
    endmillis = int(round(time.time() * 1000))
    print("Time taken: ", endmillis - stmillis)

    # Save the trained weights
    model.save("unet.h5")

    # Save model config as json
    model_json = model.to_json()
    with open("unet.json", "w") as json_file:
        json_file.write(model_json)

    # In case you wish to load your saved model
    model.load_weights("unet.h5")

    import gc
    gc.collect()

    # Load test data in X_test and y_test
    # YOUR CODE HERE
    # X_test, y_test = ...
    X_test, y_test = load_data("Test")
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(X_test.shape, y_test.shape)

    # Predict using model.predict_generator().
    # YOUR CODE HERE
    # test_data_generator = ...
    # y_pred = ...
    test_data_generator = Surface_Generator(X_test, y_test, batch_size)
    y_pred = model.predict_generator(test_data_generator)

    # y_true will have the true masks
    y_true = test_data_generator.get_all_masks()
    print("Dice coefficient on test data: ", K.get_value(dice_coef(y_true, y_pred)))

