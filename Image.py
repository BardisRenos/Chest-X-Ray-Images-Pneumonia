from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
import numpy as np

class NeuralNetwork:

    def export_data(self):
        # Create generator
        datagen = ImageDataGenerator(rescale=1./255)

        # Setting the image size 
        global IMG_SIZE
        IMG_SIZE = 250
        
        # Retrieving the data from the source file.
        train_data = datagen.flow_from_directory('C:\\Users\\Renos\\Desktop\\chest_xray\\train\\', target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=128, shuffle=True, seed=42)
        val_data = datagen.flow_from_directory('C:\\Users\\Renos\\Desktop\\chest_xray\\val\\', target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=16, shuffle=True, seed=42)
        test_data = datagen.flow_from_directory('C:\\Users\\Renos\\Desktop\\chest_xray\\test\\', target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=128, shuffle=True, seed=42)
    
        return train_data, val_data, test_data    

    def training_model(self):
        train_data, val_data, test_data = self.export_data()
        # Load the VGG model without the last layers
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

        # We can add the last part the classifier layers
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Full connected layers
        x = Dense(4096, activation='relu')(x)  # dense layer 1  with 4096 neurons
        x = Dense(1024, activation='relu')(x)  # dense layer 2  with 1024 neurons
        x = Dense(4096, activation='relu')(x)  # dense layer 3  with 4096 neurons
        
        # We change the output layer to 2 classes 
        output = Dense(2, activation='softmax')(x)

        # We can Define the new model
        model = Model(inputs=vgg_model.input, outputs=output)

        # We stop the layers to be trained
        for layer in model.layers:
            layer.trainable = False

        # We choose the parameters of the model in order to train.
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # We train the model 
        model.fit_generator(train_data, steps_per_epoch=1000, epochs=5, validation_data=val_data, validation_steps=800)

        # Lastly, we evaluate the model by appling on the test dataset
        loss = model.evaluate_generator(test_data, steps=16)

        return model, test_data
    
    def eval_model(self):
        model, test_data = self.training_model()
        test_data.reset()
        pred = model.predict_generator(test_data, steps=16, verbose=1)
        predict_class_indices = np.argmax(pred, axis=1)
        print(predict_class_indices)
        
if __name__ == '__main__':
    NN = NeuralNetwork()
    NN.eval_model()
    