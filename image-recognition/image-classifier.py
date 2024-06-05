from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow as tf
import numpy as np

# Loading the dataset
datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest', validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory('image-recognition/train', 
target_size=(256, 256), batch_size=4, class_mode='categorical', subset='training')

validation_data = datagen.flow_from_directory('image-recognition/train', 
target_size=(256, 256), batch_size=4, class_mode='categorical',subset='validation')

test_data = datagen.flow_from_directory('image-recognition/test',
target_size=(256, 256), batch_size=1, class_mode='categorical')

## Build the model
model = keras.Sequential([
    keras.Input(shape=(256, 256, 3)),
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(len(train_data.class_indices), activation="softmax")
])

learning_rate = 0.0001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

overfitting_stoppage = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

## Train the model
model.fit(train_data, epochs=30, validation_data = validation_data, callbacks=[overfitting_stoppage])

## Evaluate the trained model
test_loss, test_acc = model.evaluate(test_data, verbose=2)
print('\n Test accuracy:', test_acc)

validation_loss, validation_acc = model.evaluate(validation_data, verbose=2)
print('\n Validation accuracy:', validation_acc)

model.save('image-recognition/model2.h5')