from keras.models import Sequential, load_model
from keras.layers import Convolution2D, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(ZeroPadding2D(padding=(1,1), input_shape=(128,128,3)))
#adds extra border pixels to prevent excessive undersizing
model.add(Convolution2D(filters=32,kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(filters=64,kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(filters=64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(filters=128,kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(filters=128,kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(filters=128,kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation='sigmoid'))
    
print ("Create model successfully")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('flowers',
                                                    target_size=(128, 128),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('flowers_test',
                                                        target_size=(128,128),
                                                        batch_size=16,
                                                        class_mode='categorical')



model.fit_generator(training_set,
                         steps_per_epoch=130,
                         epochs=25,
                        validation_data=test_set,
                         validation_steps=8)

model.save("model_75.h5")
