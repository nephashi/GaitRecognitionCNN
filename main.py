import keras
from experiment_utils.gait_io import load_90_degree_gei_for_experiment1
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from layers.Conv2D121 import Conv2D121
import os

batch_size = 4
num_classes = 124
epochs = 10000
save_dir = os.path.join(os.getcwd(), 'savedd_models')
model_name = 'keras_gait_cnn.h5'

model = Sequential()
model.add(Conv2D(8, (5, 5), padding='valid',
                 input_shape=(140, 140, 1)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D121(8, (5, 5), padding='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D121(8, (5, 5), padding='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D121(8, (5, 5), padding='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(num_classes, input_shape=(200,)))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train, y_train, x_test, y_test = load_90_degree_gei_for_experiment1('Z:/DatasetB/GEI_CASIA_B/GEI_CASIA_B/gei90/', 'cl2')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])