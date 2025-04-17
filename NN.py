import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import cv2
import keras
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# dataset hoda


train_dataset_path = 'Train 60000.cdb'
test_dataset_path = 'Test 20000.cdb'
testt_dataset_path = 'RemainingSamples.cdb'
x_train, y_train = read_hoda_dataset(dataset_path=train_dataset_path,images_height=32,images_width=32)
x_test, y_test = read_hoda_dataset(dataset_path=test_dataset_path,images_height=32,images_width=32)
x_testt, y_testt = read_hoda_dataset(dataset_path=testt_dataset_path,images_height=32,images_width=32)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='dataset/Train',
    labels='inferred',
    label_mode='categorical',
    image_size=(28, 28))

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory='dataset/Test',
    labels='inferred',
    label_mode='categorical',
    image_size=(28, 28))

print(train_ds.shape, test_ds.shape)
print(x_train.shape, x_test.shape, x_testt.shape)
#
# model = models.Sequential([
#     layers.Input(shape=(256, 256, 3)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),  # این تبدیل می‌کنه به (None, something)
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
#
#
# model.compile(optimizer='adam',loss=keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
#
# # model.build(input_shape=(None,1024))
#
# model.summary()
#
#
# callbacks=keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='min')
#
# hist = model.fit(x_train,y_train, epochs=50,batch_size=128,callbacks=[callbacks],validation_data=(x_test,y_test))
#
# model.save('NN.keras')
#
# # predictions = model.predict(x_test)  # shape: (num_samples, num_classes)
# # predicted_classes = predictions.argmax(axis=1)
#
#
# loss, accuracy = model.evaluate(x_testt, y_testt)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")