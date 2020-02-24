#!/usr/bin/env python
'''
This is a modified version of Jiawen Yao MIL implementation
'''

import numpy as np
import time

from sklearn.model_selection import train_test_split

from utl import Cell_Net
from utl import vgg16_Net
import argparse

from utl.dataset import load_dataset

import glob

from utl.DataGenerator import DataGenerator

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.0005, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=100, type=int)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=False, type=int)
    parser.add_argument('--dataPath', dest='dataPath',
                        help='path to the image data',
                        default='data/cifar-10-miml', type=str)
    parser.add_argument('--runMode', dest='run_mode',
                        help='train or eval',
                        default='train', type=str)
    parser.add_argument('--network', dest='network',
                        help='networks that can be used cell_net or vgg16',
                        default='cell_net', type=str)
    parser.add_argument('--image_size', dest='image_size',
                        help='image size in pixels',
                        default=32, type=int)
    parser.add_argument('--transfer_learning', dest='transfer_learning',
                        help='use pre-trained weights',
                        default=True, type=bool)
    parser.add_argument('--layers_not_trainable', dest='layers_not_trainable',
                        help='layers to block when doing transfer learning',
                        default=17, type=int)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    args = parser.parse_args()
    return args


def generate_batch(path, labels, input_dim):
    """Loads and prepare image data and labels for training.
    Parameters
    -----------------
    path : list
        List of paths to the image folders
    labels : list
        list of labels for each image folder
    input_dim : list
        list with height and with of the images to be passe to the model
    Returns
    -----------------
    bags : list
        List of Lists containing image data, labels, and path to the image
    """
    bags = []
    for each_path_idx in range(len(path)):
        name_img = []
        img = []
        bag_label = []
        img_path = glob.glob(path[each_path_idx] + '/*.*')

        for each_img in img_path:
            img_raw = load_img(each_img, target_size=(input_dim[0], input_dim[1]))  # this is a PIL image
            img_data = img_to_array(img_raw) / 255  # this is a Numpy array with shape (3, 256, 256)
            img.append(np.expand_dims(img_data, 0))
            name_img.append(each_img.split('/')[-1])
            bag_label.append(np.expand_dims(labels[each_path_idx], 0))

        stack_img = np.concatenate(img, axis=0)
        stack_labels = np.concatenate(bag_label, axis=0)
        bags.append((stack_img, stack_labels, name_img))

    return bags


def test_eval(model, test_set, processclass):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    processclass : integer
        The class to evaluate from the multiclass
    Returns
    -----------------
    acc : dict
        Dictionary with evaluation resoults
    """
    num_test_batch = len(test_set)
    test_pred = np.zeros((num_test_batch, 1), dtype=bool)
    test_label = np.zeros((num_test_batch, 1), dtype=int)
    false_bag_predict = 0
    true_bag_predict = 0
    false_instance_predict = 0
    true_instance_predict = 0
    true_bag_instance_predict = 0
    found_path = []
    print('Processing Class: ', processclass)
    get_alpha_layer_output1 = K.function([model.layers[0].input], [model.get_layer("alpha" + str(processclass)).output])

    for ibatch, batch in enumerate(test_set):

        result = model.predict_on_batch(x=batch[0])[0].round()
        layer_outs = get_alpha_layer_output1([batch[0], 1.])

        test_label[ibatch] = batch[1][0][processclass]
        test_pred[ibatch] = result[processclass]

        if test_label[ibatch][0] == 0:
            continue

        maxidx = np.asarray(layer_outs).argmax()
        found_path.append(batch[2][maxidx])
        if batch[2][maxidx].find(str(processclass) + '_') > -1:
            true_instance_predict = true_instance_predict + 1
            if result[processclass] == test_label[ibatch]:
                true_bag_instance_predict += 1

        else:
            false_instance_predict = false_instance_predict + 1

        if result[processclass] == test_label[ibatch]:
            true_bag_predict = true_bag_predict + 1
        else:
            false_bag_predict = false_bag_predict + 1

    bag_accuracy = 0
    instance_accuracy = 0
    true_positive_accuracy = 0

    if (true_bag_predict + false_bag_predict) > 0:
        bag_accuracy = true_bag_predict / (true_bag_predict + false_bag_predict)
        instance_accuracy = true_instance_predict / (true_bag_predict + false_bag_predict)
    if (true_instance_predict + false_instance_predict) > 0:
        true_positive_accuracy = true_bag_instance_predict / true_bag_predict

    print('bag_accuracy: ', bag_accuracy)
    print('instance_accuracy: ', instance_accuracy)
    print('true_positive_accuracy: ', true_positive_accuracy)

    print(classification_report(test_label, test_pred))

    print(confusion_matrix(test_label, test_pred))

    return {"bag_accuracy": bag_accuracy, 'instance_accuracy': instance_accuracy,
            'true_positive_accuracy': true_positive_accuracy, 'found_path': found_path}


def train_eval(model, train_set, test_set, model_save_path):
    """Train the model
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    test_set : list
        A list of test set contains all test bags features and labels.
    model_save_path : string
        The model is saved with name including the dataset path
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1

    train_gen = DataGenerator(batch_size=1, shuffle=True).generate(train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(test_set)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    model_name = "saved_model/" + model_save_path + "_best.hd5"

    # Load model and continue training.
    if os.path.exists(model_name):
        print("load saved model weights")
        model.load_weights(model_name)

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_bag_accuracy', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_bag_accuracy', patience=30)

    callbacks = [checkpoint_fixed_name, EarlyStop]

    history = model.fit_generator(generator=train_gen, steps_per_epoch=len(train_set) // batch_size,
                                  epochs=args.max_epoch, validation_data=val_gen,
                                  validation_steps=len(test_set) // batch_size, callbacks=callbacks)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['bag_accuracy']
    val_acc = history.history['val_bag_accuracy']

    if not os.path.exists('results'):
        os.makedirs('results')

    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'results/loss_' + model_save_path + "_best.png"
    fig.savefig(save_fig_name)

    fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'results/acc_' + model_save_path + "_best.png"
    fig.savefig(save_fig_name)

    return model_name


def calculating_class_weights(y_true):
    """Calculates the class weight to make the training balanced
    Parameters
    -----------------
    y_true : list
        A list of labels.
    Returns
    -----------------
    weights: 2D array with weights for each class
    """
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights


def model_training(path, label, input_dim, num_classes, model_save_path, network):
    """Split data and train and evaluate the model
    Parameters
    -----------------
    path : list
        List of paths to the image folders
    labels : list
        list of labels for each image folder
    input_dim : list
        list with height and with of the images to be passe to the model
    num_classes : int
        The number of classes in the multiclass classification
    model_save_path : list
        The model is saved with name including the dataset path
    Returns
    -----------------
    acc[]: List of Evaluation accuracies
    """
    X_train, X_test, y_train, y_test = train_test_split(path, label, test_size=0.25, random_state=42)

    train_set = generate_batch(X_train, y_train, input_dim)
    test_set = generate_batch(X_test, y_test, input_dim)

    class_weight = calculating_class_weights(y_train)
    print("class weights: ", class_weight)

    if network == 'cell_net':
        model = Cell_Net.cell_net(input_dim, args, num_classes, class_weight, useMulGpu=False)
    else:
        model = vgg16_Net.vgg16_net(input_dim, args, num_classes, class_weight, useMulGpu=False)

    # train model
    t1 = time.time()
    train_eval(model, train_set, test_set, model_save_path)
    t2 = time.time()
    print('training time:', (t2 - t1) / 60.0, 'min')

    return model_eval(test_set, num_classes, model_save_path, class_weight,network)


def model_eval(test_set, num_classes, model_save_path, class_weight, network):
    """Evaluate the model one class at the time
    Parameters
    -----------------
    test_set : list
        A list of test set contains all test bags features and labels.
    model_save_path : list
        The model is saved with name including the dataset path
    class_weight : list
        2D array with weights for each class
    Returns
    -----------------
    acc[]: List of Evaluation accuracies
    """
    if network == 'cell_net':
        model = Cell_Net.cell_net(input_dim, args, num_classes, class_weight, useMulGpu=False)
    else:
        model = vgg16_Net.vgg16_net(input_dim, args, num_classes, class_weight, useMulGpu=False)

    t1 = time.time()
    model_name = "saved_model/" + model_save_path + "_best.hd5"
    print("load saved model weights")
    model.load_weights(model_name)

    acc = np.zeros((num_classes), dtype=dict)
    for num_class in range(num_classes):
        acc[num_class] = test_eval(model, test_set, num_class)

    t2 = time.time()
    print('eval time:', (t2 - t1) / 60.0, 'min')
    return acc


def print_accuracy(acc):
    """Print the evaluation accuracies
    Parameters
    -----------------
    acc[]: List of Evaluation accuracies
    """
    bag_accuracy = []
    instance_accuracy = []
    true_positive_accuracy = []

    print("| |bag_accuracy|instance_acc|true_positive_acc|")
    print("|------- |:----:|:----:|:----:|")

    for process_class in range(num_classes):
        bag_accuracy.append(acc[process_class]["bag_accuracy"])
        instance_accuracy.append(acc[process_class]["instance_accuracy"])
        true_positive_accuracy.append(acc[process_class]["true_positive_accuracy"])

        print("|class {0} | {1:.2f} | {2:.2f} | {3:.2f} |"
              .format(process_class,
                      acc[process_class]["bag_accuracy"],
                      acc[process_class]["instance_accuracy"],
                      acc[process_class]["true_positive_accuracy"]))

    print("|MEAN    | {0:.2f} | {1:.2f} | {2:.2f} |"
          .format(np.average(bag_accuracy), np.average(instance_accuracy), np.average(true_positive_accuracy)))


if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    input_dim = (args.image_size, args.image_size, 3)
    data_path = args.dataPath
    network = args.network

    dataset, label = load_dataset(data_path)
    num_classes = len(label[0])
    acc = []
    model_save_path = data_path.replace('\\', '/').split('/')[-1]
    if args.run_mode == 'train':
        acc = model_training(dataset, label, input_dim, num_classes, model_save_path, network)
    elif args.run_mode == 'eval':
        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.25, random_state=42)
        acc = model_eval(generate_batch(X_test, y_test, input_dim), num_classes, model_save_path,
                         calculating_class_weights(y_train),network)
    else:
        print("runMode should be either train or eval")

    print_accuracy(acc)
