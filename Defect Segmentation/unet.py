from __future__ import print_function

from keras.layers import Conv2DTranspose, Dropout, Activation, BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.models import Model


# Create a 2D convolution block. We will use multiple instances of this block to build our U-net model
# This block will contain two layers.
# Each layer will be a Convolution operation followed by batch normalization with relu activation
def conv2d_block(input_tensor, n_filters, kernel_size):
    # first layer
    # Create a Conv2D layer with n_filters and a kernel of dimension : kernel_size x kernel_size.
    # Use same padding and he_normal initializer
    # YOUR CODE HERE
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)

    # add a BatchNormalization layer
    # YOUR CODE HERE
    x = BatchNormalization()(x)

    # Add a relu non-linearity (keras.layers.Activation)
    # YOUR CODE HERE
    x = Activation("relu")(x)

    # second layer
    # repeat the above steps (Conv + batchnorm + relu) taking the output of relu layer as input for this convolutional layer
    # YOUR CODE HERE
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # return the output tensor
    return x


def get_unet_model(n_filters=16, dropout_prob=0.5, kernel_size=3):
    input_img = Input((512, 512, 1))

    # contracting path
    # create a convolutional block with input_img as the input tensor and n_filters
    # YOUR CODE HERE
    # c1 = ...
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size)
    # apply a 2d maxpooling with a pool size of 2x2
    # YOUR CODE HERE
    # p1 = ...
    p1 = MaxPooling2D((2, 2))(c1)
    # add a dropout. Since this the input, set the dropout rate to 0.2
    # YOUR CODE HERE
    # p1 = ...
    p1 = Dropout(dropout_prob * 0.5)(p1)

    # create another convolutional block. this time use p1 as input tensor and twice the n_filters
    # repeat the same maxpool and dropout but set dropout rate to dropout_prob this time
    # YOUR CODE HERE
    # c2 = ...
    # p2 = ...
    # p2 = ...
    c2 = conv2d_block(p1, n_filters * 2, kernel_size)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout_prob)(p2)

    # create another block with maxpool and dropout with 4 x n_filters
    # YOUR CODE HERE
    # c3 = ...
    # p3 = ...
    # p3 = ...
    c3 = conv2d_block(p2, n_filters * 4, kernel_size)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout_prob)(p3)

    # create another block with maxpool and dropout with 8 x n_filters
    # YOUR CODE HERE
    # c4 = ...
    # p4 = ...
    # p4 = ...
    c4 = conv2d_block(p3, n_filters * 8, kernel_size)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout_prob)(p4)

    # This is the layer where we combine the contractive and expansive paths
    # create a convolutional block with 16 x n_filters. No pooling/dropout this time
    # YOUR CODE HERE
    # c5 = ...
    c5 = conv2d_block(p4, n_filters * 16, kernel_size)

    # Expansive path

    # We will create a similar structure as the contracting path but instead of
    # convolutional operation, we will use Deconvolution operations

    # Create a Conv2DTranspose layer (deconvolution) with 8 x n_filters, kernel_size,
    # 2x2 strides and same padding
    # YOUR CODE HERE
    # u6 = ...
    u6 = Conv2DTranspose(n_filters * 8, kernel_size, strides=(2, 2), padding='same')(c5)
    # Concatenate u6 and c4 using keras.layers.concatenate
    # YOUR CODE HERE
    # u6 = ...
    u6 = concatenate([u6, c4])
    # dropout
    # YOUR CODE HERE
    # u6 = ...
    u6 = Dropout(dropout_prob)(u6)
    # create a convolutional block with 8 x n_filters
    # YOUR CODE HERE
    # c6 = ...
    c6 = conv2d_block(u6, n_filters * 8, kernel_size)

    # Create a similar module as previous, deconv, concatenate, dropout, conv2d_block
    # Please ensure that the number of filters you use match the n_filters of
    # the layer you are concatenating with
    # YOUR CODE HERE
    # u7 = ...
    # u7 = ...
    # u7 = ...
    # c7 = ...
    u7 = Conv2DTranspose(n_filters * 4, kernel_size, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout_prob)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size)

    # Create a similar module as previous, deconv, concatenate, dropout, conv2d_block
    # YOUR CODE HERE
    # u8 = ...
    # u8 = ...
    # u8 = ...
    # c8 = ...
    u8 = Conv2DTranspose(n_filters * 2, kernel_size, strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout_prob)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size)

    # Create a similar module as previous, deconv, concatenate, dropout, conv2d_block
    # YOUR CODE HERE
    # u9 = ...
    # u9 = ...
    # u9 = ...
    # c9 = ...
    u9 = Conv2DTranspose(n_filters * 1, kernel_size, strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout_prob)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size)

    # apply a 1x1 convolution on c9 to get an output with a single channel
    # This is the final model output. We want the pixel values in the mask to be
    # either 0 or 1. Choose an activation function which can give values in that
    # range.
    # YOUR CODE HERE
    # outputs = ...
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model