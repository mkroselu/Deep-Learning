# Fundamentals of Deep Learning 

## Assessment

Congratulations on going through today's course! Hopefully, you've learned some valuable skills along the wayand had fun doing it. Now it's time to put those skills to the test. In this assessment, you will train a new modelthat is able to recognize fresh and rotten fruit. You will need to get the model to a validation accuracy of 92% in order to pass the assessment, though we challenge you to do even better if you can. You will have to use the skills that you learned in the previous exercises. Specifically, we suggest using some combination of transferlearning, data augmentation, and fine tuning. Once you have trained the model to be at least 92% accurate onthe validation dataset, save your model, and then assess its accuracy. Let's get started!

## The Dataset 

In this exercise, you will train a model to recognize fresh and rotten fruits. The dataset comes from Kaggle (https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification), a great place to go if you're interested in starting a project after this class. The dataset structure is in the data/fruits folder. There are 6 categoriesof fruits: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas. This will mean that your model will require an output layer of 6 neurons to do the categorization successfully. You'll also need to compile the model with
categorical_crossentropy, as we have more than two categories.

## Load ImageNet Base Model 

We encourage you to start with a model pretrained on ImageNet. Load the model with the correct weights, set aninput shape, and choose to remove the last layers of the model. Remember that images have three dimensions: a height, and width, and a number of channels. Because these pictures are in color, there will be three channels for red, green, and blue. We've filled in the input shape for you. This cannot be changed or the assessment will fail.


```python
from tensorflow import keras

base_model = keras.applications.VGG16(
    weights = "imagenet", 
    input_shape = (224, 224, 3),
    include_top = False)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58889256/58889256 [==============================] - 3s 0us/step
    

## Freeze Base Model

Next, we suggest freezing the base model. This is done so that all the learning from the ImageNet dataset does not get destroyed in the initial training.


```python
# Freeze base model
base_model.trainable = False
```

## Add Layers to Model

Now it's time to add layers to the pretrained model. Pay close attention to the last dense layer and make sure it has the correct number of neurons to classify the different types of fruit.


```python
# Create inputs with correct shape
inputs = keras.Input(shape = (224, 224,3))
x = base_model(inputs, training = False) 

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(1, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)
```


```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                     
     global_average_pooling2d (G  (None, 512)              0         
     lobalAveragePooling2D)                                          
                                                                     
     dense (Dense)               (None, 1)                 513       
                                                                     
    =================================================================
    Total params: 14,715,201
    Trainable params: 513
    Non-trainable params: 14,714,688
    _________________________________________________________________
    

## Compile Model

Now it's time to compile the model with loss and metrics options. Remember that we're training on a number of different categories, rather than a binary classification problem.


```python
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])
```

## Augment the Data

If you'd like, try to augment the data to improve the dataset. There is also documentation for the
Keras ImageDataGenerator class (https://keras.io/api/preprocessing/image/#imagedatagenerator-class). This step is optional, but it may be helpful to get to 92% accuracy. 


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
    rotation_range = 10, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image
    width_shift_range = 0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.1, # randomly shift images vertically (fraction of total height)
    horizontal_flip = True, # randomly flip images horizontally
    vertical_flip = False, # Don't randomly flip images vertically
)

datagen_valid = ImageDataGenerator(samplewise_center = True)
```

## Load Dataset

Now it's time to load the train and validation datasets. Pick the right folders, as well as the right `target_size` of the images (it needs to match the height and width input of the model you've created).


```python
# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "data/fruits/train/",
    target_size = (224, 224),
    color_mode = "rgb", 
    class_mode = "categorical",
)

# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    "data/fruits/valid/",
    target_size = (224, 224),
    color_mode = "rgb",
    class_mode = "categorical",
)
```

## Train the Model 

Time to train the model! Pass the `train` and `valid` iterators into the `fit` function, as well as setting the desired number of epochs.


```python
model.fit(train_it, 
          validation_data = valid_it,
          steps_per_epoch = train_it.samples/train_it.batch_size,
          validation_steps = valid_it.samples/valid_it.batch_size,
          epochs = 20
)
```

## Unfreeze Model for Fine Tuning

If you have reached 92% validation accuracy already, this next step is optional. If not, we suggest fine tuning the model with a very low learning rate.


```python
# Unfreeze the base model
base_model.trainable = True

# Compile the model with a low learning rate
model.compile(optimizer = keras.optimizers.RMSprop(learning_rate = .00001), 
              loss = 'categorical_crossentropy', metrics = ['accuracy'])
```


```python
model.fit(train_it,
          validation_data = valid_it, 
          steps_per_epoch = train_it.samples/train_it.batch_size,
          validation_steps = valid_it.samples/valid_it.batch_size,
          epochs = 20)
```

## Evaluate the Model 

Hopefully, you now have a model that has a validation accuracy of 92% or higher. If not, you may want to go back and either run more epochs of training, or adjust your data augmentation.

Once you are satisfied with the validation accuracy, evaluate the model by executing the following cell. The evaluate function will return a tuple, where the first value is your loss, and the second value is your accuracy. To pass, the model will need to have an accuracy value of 92% or higher.


```python
model.evaluate(valid_it, steps = valid_it.samples/valid_it.batch_size)
```

## Run the Assessment

To assess your model run the following two cells.

<b> NOTE </b>: `run_assessment` assumes your model is named `model` and your validation data iterator is called `valid_it`. If for any reason you have modified these variable names, please update the names of thearguments passed to `run_assessment`.


```python
from run_assessment import run_assessment
```


```python
run_assessment(model, valid_it)
```


```python

```
