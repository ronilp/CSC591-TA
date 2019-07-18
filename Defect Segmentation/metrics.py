from keras import backend as K


# Accuracy metric
def defect_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=1)


# Dice Coefficient metric
def dice_coef(y_true, y_pred):
    smooth = 1
    # YOUR CODE HERE
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Dice Coefficient loss
def dice_coef_loss(y_true, y_pred):
    # YOUR CODE HERE
    return 1-dice_coef(y_true, y_pred)