#!/usr/bin/env python
'''
This is a modified version of Jiawen Yao MIL implementation
'''

import numpy as np
import time
from utl import Cell_Net
from random import shuffle
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
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
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
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def generate_batch(path, input_dim, labels):
    bags = []
    for each_path_idx in range(len(path)):
        name_img = []
        img = []
        img_path = glob.glob(path[each_path_idx] + '/*.*')
        num_ins = len(img_path)

        each_path = path[each_path_idx].replace('\\', '/')  # support for ms windows paths
        if labels:
            label = labels[each_path_idx]
        else:
            label = int(each_path.split('/')[-2])

        if label == 1:
            curr_label = np.ones(num_ins, dtype=np.uint8)
        else:
            curr_label = np.zeros(num_ins, dtype=np.uint8)

        for each_img in img_path:
            img_raw = load_img(each_img, target_size=(input_dim[0], input_dim[1]))  # this is a PIL image
            img_data = img_to_array(img_raw) / 255  # this is a Numpy array with shape (3, 256, 256)
            img.append(np.expand_dims(img_data, 0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label, name_img))

    return bags


def get_train_valid_Path(Train_set, train_percentage=0.8):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    indexes = np.arange(len(Train_set))
    shuffle(indexes)

    num_train = int(train_percentage * len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def test_eval(model, test_set, processclass):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    test_pred = np.zeros((num_test_batch, 1), dtype=bool)
    test_label = np.zeros((num_test_batch, 1), dtype=int)
    false_bag_predict = 0
    true_bag_predict = 0
    false_instance_predict = 0
    true_instance_predict = 0
    true_bag_instance_predict = 0
    found_path = []
    get_alpha_layer_output = K.function([model.layers[0].input], [model.get_layer("alpha").output])
    for ibatch, batch in enumerate(test_set):

        result = model.predict_on_batch(x=batch[0]).round()  # , y=batch[1])

        layer_outs = get_alpha_layer_output([batch[0], 1.])

        test_label[ibatch] = batch[1][0]
        test_pred[ibatch] = result

        if test_label[ibatch] == 0:
            continue

        maxidx = np.asarray(layer_outs).argmax()
        found_path.append(batch[2][maxidx])
        if batch[2][maxidx].find(str(processclass) + '_') > -1:
            true_instance_predict = true_instance_predict + 1
            if result == test_label[ibatch]:
                true_bag_instance_predict += 1

        else:
            false_instance_predict = false_instance_predict + 1

        if result == test_label[ibatch]:
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


def train_eval(model, train_set, data_path=None):
    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1
    model_train_set, model_val_set = get_train_valid_Path(train_set, train_percentage=0.9)

    train_gen = DataGenerator(batch_size=1, shuffle=True).generate(model_train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_val_set)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    model_name = "saved_model/" + data_path + "_best.hd5"

    # Load model and continue training.
    if os.path.exists(model_name):
        print("load saved model weights")
        model.load_weights(model_name)

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_bag_accuracy', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_bag_accuracy', patience=30)

    callbacks = [checkpoint_fixed_name, EarlyStop]

    history = model.fit_generator(generator=train_gen, steps_per_epoch=len(model_train_set) // batch_size,
                                  epochs=args.max_epoch, validation_data=val_gen,
                                  validation_steps=len(model_val_set) // batch_size, callbacks=callbacks,
                                  class_weight={0: 1, 1: 3})

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
    save_fig_name = 'results/loss_' + data_path + "_best.png"
    fig.savefig(save_fig_name)

    fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'results/acc_' + data_path + "_best.png"
    fig.savefig(save_fig_name)

    return model_name


def model_training(input_dim, dataset, label=None, data_path=None, process_class=None):
    train_bags = dataset['train']
    test_bags = dataset['test']

    if label:
        train_set = generate_batch(train_bags, input_dim, label['train'])
        test_set = generate_batch(test_bags, input_dim, label['test'])
    else:
        # convert bag to batch
        train_set = generate_batch(train_bags, input_dim)
        test_set = generate_batch(test_bags, input_dim)

    model = Cell_Net.cell_net(input_dim, args, useMulGpu=False)
    # train model
    t1 = time.time()

    model_name = train_eval(model, train_set, data_path)

    print("load saved model weights")
    model.load_weights(model_name)

    test_res = test_eval(model, test_set, process_class)

    t2 = time.time()

    print('run time:', (t2 - t1) / 60.0, 'min')

    return test_res


def model_eval(data_path, test_set, process_class):
    # train model
    model = Cell_Net.cell_net(input_dim, args, useMulGpu=False)

    t1 = time.time()
    model_name = "saved_model/" + data_path + "_best.hd5"
    print("load saved model weights")
    model.load_weights(model_name)

    test_res = test_eval(model, test_set, process_class)

    t2 = time.time()
    print('run time:', (t2 - t1) / 60.0, 'min')
    return test_res


if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    input_dim = (32, 32, 3)
    data_path = args.dataPath
    n_folds = 4
    num_classes = 10

    acc = np.zeros((num_classes), dtype=dict)

    for process_class in range(num_classes):
        dataset, label = load_dataset(dataset_path=data_path, n_folds=n_folds, rand_state=42,
                                      numClasses=process_class)
        print('class=', process_class)
        model_save_path = data_path.replace('\\', '/').split('/')[-1] + "_class" + str(process_class)
        if args.run_mode == 'train':
            acc[process_class] = model_training(input_dim, dataset[0], label[0], model_save_path, process_class)
        if args.run_mode == 'eval':
            test_set_eval = generate_batch(dataset[0]['test'], input_dim, label[0]['test'])
            acc[process_class] = model_eval(model_save_path, test_set_eval, process_class)

    bag_accuracy = []
    instance_accuracy = []
    true_positive_accuracy = []

    for process_class in range(num_classes):
        bag_accuracy.append(acc[process_class]["bag_accuracy"])
        instance_accuracy.append(acc[process_class]["instance_accuracy"])
        true_positive_accuracy.append(acc[process_class]["true_positive_accuracy"])

        print("class {0} bag_accuracy {1:.2f} instance_acc {2:.2f} true_positive_acc {3:.2f}"
              .format(process_class,
                      acc[process_class]["bag_accuracy"],
                      acc[process_class]["instance_accuracy"],
                      acc[process_class]["true_positive_accuracy"]))

    print("MEAN bag_accuracy {0:.2f} instance_acc {1:.2f} true_positive_acc {2:.2f}"
          .format(np.average(bag_accuracy), np.average(instance_accuracy), np.average(true_positive_accuracy)))
