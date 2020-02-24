from keras.utils import multi_gpu_model, get_file
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, multiply, concatenate
from .metrics import bag_accuracy, get_weighted_loss, bag_loss
from .custom_layers import Mil_Attention, Last_Sigmoid


def vgg16_net(input_dim, args, num_classes, class_weight=None, useMulGpu=False):
    """VGG16 network architecture, as defined in:
     https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
     By setting transfer_learning pretrained weights from IMAGENET is loaded
     layers_not_trainable should initially be set to 17 to train the fully conencted and attention layer
    Parameters
    -----------------
    input_dim : tuple
        tuple with the image dimensions, VGG pre-trained weights are trained using 224 X 224
    args : list
        Command line arguments controlling network parameters
    num_classes : int
        The number of classes in the output
    class_weight : list
        2D array with weights for each class
    useMulGpu : bool
        Train on 2 gpu's
    Returns
    -----------------
    acc[]: List of Evaluation accuracies
    """
    lr = args.init_lr
    weight_decay = args.init_lr
    layers_not_trainable = args.layers_not_trainable
    transfer_learning = args.transfer_learning

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(data_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if transfer_learning:
        base_model = Model(input=[data_input], output=[x])
        WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                               'releases/download/v0.1/'
                               'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='6d6bbae143d832006294945121d1f1fc')
        base_model.load_weights(weights_path)

        x = Flatten(name='flatten')(base_model.output)
    else:
        x = Flatten(name='flatten')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)
    fc2 = Dense(4096, activation='relu', name='fc2')(x)

    output_layers = []
    for i in range(num_classes):
        alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha' + str(i),
                              use_gated=args.useGated)(fc2)
        x_mul = multiply([alpha, fc2])

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid' + str(i))(x_mul)
        output_layers.append(out)

    x_out = concatenate(output_layers)

    model = Model(inputs=[data_input], outputs=x_out)

    print(model.summary())

    for layer in model.layers[:layers_not_trainable]:
        layer.trainable = False

    for layer in model.layers:
        print(layer.name)
        print(layer.trainable)

    if class_weight.any():
        loss = get_weighted_loss(class_weight)
    else:
        loss = bag_loss

    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=loss,
                               metrics=[bag_accuracy])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=loss,
                      metrics=[bag_accuracy])
        parallel_model = model
    return parallel_model
