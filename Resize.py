import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow import keras
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset


train_dataset_path = 'Train 60000.cdb'
test_dataset_path = 'Test 20000.cdb'
testt_dataset_path = 'RemainingSamples.cdb'
x_train, y_train = read_hoda_dataset(dataset_path=train_dataset_path,images_height=28,images_width=28)
x_test, y_test = read_hoda_dataset(dataset_path=test_dataset_path,images_height=28,images_width=28)
x_testt, y_testt = read_hoda_dataset(dataset_path=testt_dataset_path,images_height=28,images_width=28
                                     )

model = Sequential([
    layers.Input(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.save('cnn.keras')


callbacks=keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,verbose=1,mode='min')

hist = model.fit(x_train,y_train, epochs=50,batch_size=128,callbacks=[callbacks],validation_data=(x_test,y_test))

loss, accuracy = model.evaluate(x_testt, y_testt)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")