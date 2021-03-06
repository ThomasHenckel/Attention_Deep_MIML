from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, multiply, concatenate
from .metrics import bag_accuracy, get_weighted_loss
from .custom_layers import Mil_Attention, Last_Sigmoid


def cell_net(input_dim, args, num_classes, class_weight, useMulGpu=False):
    lr = args.init_lr
    weight_decay = args.init_lr

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1 = Conv2D(36, kernel_size=(4, 4), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(48, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv2 = MaxPooling2D((2, 2))(conv2)
    x = Flatten()(conv2)

    fc1 = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay), name='fc1')(x)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay), name='fc2')(fc1)
    fc2 = Dropout(0.5)(fc2)

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

    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=get_weighted_loss(class_weight),
                               metrics=[bag_accuracy])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=get_weighted_loss(class_weight),
                      metrics=[bag_accuracy])
        parallel_model = model

    return parallel_model
