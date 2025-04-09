import tensorflow as tf
from tensorflow.keras import layers
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sn

#لود دیتاست
X_train, y_train = read_hoda_dataset(dataset_path='Train 60000.cdb',
                                     images_height=32,
                                     images_width=32,
                                     one_hot=False,
                                     reshape=True)

X_test, y_test = read_hoda_dataset(dataset_path='Test 20000.cdb',
                                   images_height=32,
                                   images_width=32,
                                   one_hot=False,
                                   reshape=True)

X_remaining, y_remaining = read_hoda_dataset(dataset_path='RemainingSamples.cdb',
                                             images_height=32,
                                             images_width=32,
                                             one_hot=False,
                                             reshape=True)
#پلات 3 نمونه
fig = plt.figure(figsize=(16, 3))
fig.add_subplot(1, 3, 1)
plt.title('Y_train[ 0 ] = ' + str(y_train[0]))
plt.imshow(X_train[0].reshape([32, 32]), cmap='gray')

fig.add_subplot(1, 3, 2)
plt.title('Y_test[ 0 ] = ' + str(y_test[0]))
plt.imshow(X_test[0].reshape([32, 32]), cmap='gray')

fig.add_subplot(1, 3, 3)
plt.title('y_remaining[ 0 ] = ' + str(y_remaining[0]))
plt.imshow(X_remaining[0].reshape([32, 32]), cmap='gray')

plt.show()



#ساختار مدل
model = tf.keras.models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='softmax')
])

#کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#فیت مدل
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

#ارزیابی روی تست
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\n🎯 دقت مدل روی تست: {test_acc:.4f}')







