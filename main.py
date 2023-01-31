from tensorflow import keras
from keras import layers
# from kerastuner.tuners import RandomSearch
import numpy as np
# import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import IPython

####################
### test alexnet
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
import keras.losses
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('PS') #prevent import error due to venv
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
keras.utils.set_random_seed(
    42
)

class_map = {0: 'other',
             1: 'crater',
             2: 'dark_dune',
             3: 'streak',
             4: 'bright_dune',
             5: 'impact',
             6: 'edge'}

file = open('C:/Users/quang/PycharmProjects/finalProj/hirise-map-proj/labels-map-proj.txt', 'r')
lines = [line.strip() for line in file.readlines()]
file.close()

train, not_train = train_test_split(lines, train_size=0.7)
val, test = train_test_split(not_train, test_size=0.5)

img_path = 'C:/Users/quang/PycharmProjects/finalProj/hirise-map-proj/map-proj/'

# def get_set(lines):
#   images = []
#   labels = []
#
#   for line in tqdm(lines):
#     filename, label = line.split(' ')
#     img = Image.open(img_path + filename)
#     img = np.array(img)
#     images.append(img)
#
#     encoding = [0 for _ in range(7)]
#     encoding[int(label)] = 1
#     labels.append(encoding)
def get_set(lines):
  images = []
  labels = []

  for line in tqdm(lines):
    filename, label = line.split(' ')
    img = Image.open(img_path + filename)
    img = np.array(img)
    images.append(img)

    encoding = [0 for _ in range(7)]
    encoding[int(label)] = 1
    labels.append(encoding)

    img = img/255.0
    # plt.imshow(img, cmap=plt.get_cmap('gray'))
    # plt.show()
    img = img.reshape(1, 227, 227, 1)
    # datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    # datagen.fit(img)
    # addImg = datagen.flow(img, batch_size=9)
    # i = 0
    # for img_batch in datagen.flow(img, batch_size=9):
    #     for img in img_batch:
    #         img = img.reshape(227,227)
    #         images.append(img)
    #         labels.append(encoding)
    #         # plt.subplot(330 + 1 + i)
    #         # plt.imshow(img, cmap=plt.get_cmap('gray'))
    #         i = i + 1
    #     if i >= 9:
    #         break

    # plt.show()
  images = np.array(images)
  labels = np.array(labels)

  return images, labels



print("Importing training set...")
train_images, train_labels = get_set(train)

print()
print("Importing validation set...")
val_images, val_labels = get_set(val)

print()
print("Importing test set...")
test_images, test_labels = get_set(test)

train_images = train_images/255.0
val_images = val_images/255.0
test_images = test_images/255.0

# plt.figure(figsize=(10,10))
# for i in range(15):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary_r)
#     plt.xlabel(class_map[train_labels[i]])
# plt.show()

# def plot_class_dist(labels, title):
#   classes, counts = np.unique(labels, return_counts=True)
#   classes = [class_map[label] for label in classes]
#   normalized_counts = (counts/counts.sum())
#   plt.bar(classes, normalized_counts)
#   plt.title(title)
#   plt.xticks(rotation=90)
#
#
# plt.figure(figsize=(10,2))
# plt.subplot(1,3,1)
# plot_class_dist(train_labels, 'Training')
# plt.subplot(1,3,2)
# plot_class_dist(val_labels, 'Validation')
# plt.subplot(1,3,3)
# plot_class_dist(test_labels, 'Testing')
# plt.show()

train_images = train_images.reshape((train_images.shape[0],227,227,1))
val_images = val_images.reshape((val_images.shape[0],227,227,1))
test_images = test_images.reshape((test_images.shape[0],227,227,1))

print(train_images.shape, val_images.shape, test_images.shape)


# def build_model(hp):
#     model = keras.Sequential()
#
#     # convolutional / max pooling layers
#     # For first layer, only make large stride of 4 possible if filter size is 11x11
#     layer_1_filtersize = hp.Choice('filtersize_1', values=[3, 5, 11])
#     if layer_1_filtersize == 11:
#         possible_strides = [1, 2, 4]
#     else:
#         possible_strides = [1, 2]
#     model.add(layers.Conv2D(hp.Choice('filters_1', values=[32, 64, 96, 128]),
#                             layer_1_filtersize,
#                             strides=hp.Choice('strides_1', values=possible_strides),
#                             activation='relu',
#                             input_shape=train_images.shape[1:]))
#
#     model.add(layers.MaxPooling2D((2, 2)))
#
#     # Up to 4 additional conv layers
#     for i in range(hp.Int('num_conv_layers', 1, 4)):
#         model.add(layers.Conv2D(hp.Choice('filters_' + str(i + 2), values=[32, 64, 96, 128]),
#                                 hp.Choice('filtersize_' + str(i + 2), values=[3, 5]),
#                                 activation='relu'))
#
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#
#     # fully connected layers (up to four)
#     for i in range(hp.Int('num_dense_layers', 1, 4)):
#         model.add(layers.Dense(units=hp.Int('dense_units_' + str(i + 1),
#                                             min_value=64,
#                                             max_value=512,
#                                             step=64),
#                                activation='relu'))
#         if hp.Choice('dropout_dense_' + str(i + 1), values=[True, False], default=False):
#             model.add(layers.Dropout(0.5))
#
#             # output softmax layer
#     model.add(layers.Dense(7, activation='softmax'))
#
#     # learning rate & optimizer
#     learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001])
#
#     if hp.Choice('optimizer', values=['adam', 'rmsprop']) == 'adam':
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate),
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
#     else:
#         model.compile(optimizer=keras.optimizers.RMSprop(learning_rate),
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
#
#     return model
#
# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=30,
#     executions_per_trial=3,
#     overwrite=True,
#     project_name='mars_trials')
#
# class ClearTrainingOutput(keras.callbacks.Callback):
#   def on_train_end(*args, **kwargs):
#     IPython.display.clear_output(wait = True)
#
# tuner.search(train_images, train_labels,
#              validation_data=(val_images, val_labels),
#              epochs=30,
#              callbacks=[ClearTrainingOutput()])
#
# best_hp = tuner.get_best_hyperparameters()[0]
#
# best_model = tuner.hypermodel.build(best_hp)
# best_model.summary()
#
# for i in range(best_hp.get('num_conv_layers')+1):
#   filter_size = best_hp.get('filtersize_' + str(i+1))
#   print('Filter Size (conv layer {}): {}x{}'.format(i+1, filter_size, filter_size))
# print('Stride (conv layer 1): ', best_hp.get('strides_1'))
# print('Learning Rate: ', best_hp.get('learning_rate'))
# print('Optimizer: ', best_hp.get('optimizer'))
#
# history = best_model.fit(train_images, train_labels, epochs=30,
#                 validation_data=(val_images, val_labels),
#                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)])
#
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['training', 'validation'], loc='upper left')
#
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['training', 'validation'], loc='upper right')
#
# plt.show()
#
# test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)
# print(test_accuracy)





def alexnet_cnn(train_images, train_labels, test_images, test_labels, val_images, val_labels):
    print("about to train alexnet")
    np.random.seed(1000)

    # data_augmentation = Sequential([
    #     layers.RandomFlip("horizontal_and_vertical"),
    #     layers.RandomRotation(0.2),
    # ])
    # datagen = ImageDataGenerator(rotation_range=30, horizontal_flip=0.5)
    # datagen.fit(train_images)
    # i=0
    # for img_batch in datagen.flow(train_images, batch_size=9):
    #     for img in img_batch:
    #         plt.subplot(330 + 1 + i)
    #         plt.imshow(img)
    #         i = i + 1
    #         plt.show()
    #     if i >= 32:
    #         break
    # i=0
    # batch_size= 9
    # for img_batch in datagen.flow(train_images, batch_size=9):
    #     for img in img_batch:
    #         plt.subplot(330 + 1 + i)
    #         plt.imshow(img)
    #         i = i + 1
    #     if i >= batch_size:
    #         break
    model = Sequential()
    # model.add(layers.RandomFlip("horizontal_and_vertical"))
    # model.add(layers.RandomRotation(0.2))
    # model.add(layers.RandomZoom)

    model.add(Conv2D(filters=96, input_shape=(227,227,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(227*227*1,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4096, input_shape=(227 * 227 * 1,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # plot_model(model, to_file='AlexNet.png', show_shapes=True, show_layer_names=True)
    # model.summary()

    # model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
    # history = model.fit(train_images, train_labels, batch_size=32, epochs=5)
    history = model.fit(train_images, train_labels, epochs=30, batch_size=32,
                             validation_data=(val_images, val_labels))
                        # ,
                        #      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)])


    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print("Final loss was {}.\nAccuracy of model was {}".format(test_loss, test_accuracy))

    # predictions = model.predict(test_images)
    # pred_labels = predictions.argmax(axis=1)
    #
    # cm = confusion_matrix(test_labels, pred_labels, normalize='true')
    # display_labels = np.unique(np.concatenate((test_labels, pred_labels)))
    # display_labels = [class_map[label] for label in display_labels]
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    # disp = disp.plot(cmap='Blues', xticks_rotation='vertical')
    # plt.show()


alexnet_cnn(train_images, train_labels, test_images, test_labels, val_images, val_labels)