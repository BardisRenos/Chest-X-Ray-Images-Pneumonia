# Chest-X-Ray-Images-Pneumonia

## Introduction

Artificial intelligence (AI) has the potential to revolutionize disease diagnosis and management by performing classification difficult for human experts and by rapidly reviewing immense amounts of images. Despite its potential, clinical interpretability and feasible preparation of AI remains challenging.

This repository contains a example of transfer learning method, based on X-ray images. The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse “interstitial” pattern in both lungs.

<p align="center"> 
<img src="https://github.com/BardisRenos/Chest-X-Ray-Images-Pneumonia-/blob/master/figs6.jpg" width="400" height="200" style=centerme>
</p>

### Retrieving the data from the source. 

This example uses the library [ImageDataGenerator](https://keras.io/preprocessing/image/) of Keras. The data is already splitted into three categories, ***train data***, ***test data*** and ***validate data***. 

```python
  def export_data(self):
    # Create generator with a parameter of rescaling the pixel values. 
    datagen = ImageDataGenerator(rescale=1./255)

    # Setting the image size 
    global IMG_SIZE
    IMG_SIZE = 250

    # Retrieving each data sets from the source file.
    train_data = datagen.flow_from_directory('C:\\Users\\user\\Desktop\\chest_xray\\train\\', target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=128, shuffle=True, seed=42)
    val_data = datagen.flow_from_directory('C:\\Users\\user\\Desktop\\chest_xray\\val\\', target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=16, shuffle=True, seed=42)
    test_data = datagen.flow_from_directory('C:\\Users\\user\\Desktop\\chest_xray\\test\\', target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=128, shuffle=True, seed=42)

    return train_data, val_data, test_data 
```

### Creating the Deep Neural Network model

The below model is a pre trained model (VGG-16). In this model has changed the input layer and the last part of the calssification layer.  

```python
def training_model(self):
    train_data, val_data, test_data = self.export_data()
    # Load the VGG model without the last layers
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # We stop the layers to be trained
    for layer in model.layers:
        layer.trainable = False

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

    # We choose the parameters of the model in order to train.
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # We train the model 
    model.fit_generator(train_data, steps_per_epoch=1000, epochs=5, validation_data=val_data, validation_steps=800)

    # Lastly, we evaluate the model by appling on the test dataset
    loss = model.evaluate_generator(test_data, steps=16)
```

### Evaluate the model

```python
  def eval_model(self):
      model, test_data = self.training_model()
      test_data.reset()
      pred = model.predict_generator(test_data, steps=16, verbose=1)
      predict_class_indices = np.argmax(pred, axis=1)
      print(predict_class_indices)
```
