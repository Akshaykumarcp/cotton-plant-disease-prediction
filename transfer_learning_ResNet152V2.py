# TEST GPU SET UP via commands

'''
 run commands below:
 1. nvcc -V --> nvidia cuda compiler driver
 2. nvidia-smi --> about GPU

cudNN version can be found at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include\cudnn.h as below:

#define CUDNN_MAJOR 7
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 5
'''

# TEST GPU SET UP via tendorflow

import tensorflow as tf

# check tf version
tf.__version__

# is cuda installed ?
tf.test.is_built_with_cuda()

# test whether GPU is available
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# physical_device name
tf.config.list_physical_devices('GPU')

# number of GPU's available
len(tf.config.experimental.list_physical_devices('GPU'))

# code to confirm tensorflow is using GPU
tf.config.experimental.list_physical_devices('GPU')

# CONFIG GPU

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# avoid using 100% of GPU, else GPU overclock.
config = ConfigProto()
# use 50% of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# BEGIN THE PROGRAM

# import the lib's

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet50 import preprocess_input
from glob import glob

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'dataset/train'
valid_path = 'dataset/test'

# use imagenet weights i,e download weight file from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5
resnet = ResNet152V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# not to train existing weights
for layer in resnet.layers:
    layer.trainable = False

# get no of output classes
total_no_of_classes = glob('dataset/train/*')

# flatten all the layers
x = Flatten()(resnet.output)

output_layer = Dense(len(total_no_of_classes), activation='softmax')(x)

# create object for the model
resnet_model = Model(inputs=resnet.input, outputs=output_layer)

# show model architecture
resnet_model.summary()

# inform model about cost and optimization method to use
resnet_model.compile(
    loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

# utilize 'Image Data Generator' for importing images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# give same target size as input size for the images
training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# run the resnet model
result = resnet_model.fit(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt

# plot the loss
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss_ResNet152V2')

# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc_ResNet152V2')

# save resnet50 model to local
resnet_model.save('resnet152V2_model.h5')

# prediction for testset
predict_new = resnet_model.predict(test_set)

predict_new

import numpy as np
# take argmax on testset i,e on all images
predict_new = np.argmax(predict_new, axis=1)
predict_new

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# load model from local
model=load_model('resnet152V2_model.h5')

# load new image
new_image = image.load_img('dataset/test/fresh cotton leaf/d (378).jpg',target_size=(224,224))

# convert PIL image type to array
new_image_array = image.img_to_array(new_image)

new_image_array.shape

# normalize
new_image_array=new_image_array/255

import numpy as np
# add additional dim for input of the NN
new_image_array=np.expand_dims(new_image_array,axis=0)
# https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras/47556342
new_image_array_with_new_added_dim =preprocess_input(new_image_array)
new_image_array_with_new_added_dim.shape

# predict new image
model.predict(new_image_array_with_new_added_dim)

# argmax on predicted image
maxVoter=np.argmax(model.predict(new_image_array_with_new_added_dim), axis=1)

maxVoter==2
# true

'''
61/61 [==============================] - 24s 395ms/step - accuracy: 0.8324 - loss: 1.2957 - val_loss: 0.3694 - val_accuracy: 0.9444
Epoch 2/20
61/61 [==============================] - 20s 332ms/step - accuracy: 0.9313 - loss: 0.4453 - val_loss: 0.3510 - val_accuracy: 0.8889
Epoch 3/20
61/61 [==============================] - 21s 340ms/step - accuracy: 0.9441 - loss: 0.3790 - val_loss: 0.6043 - val_accuracy: 0.9444
Epoch 4/20
61/61 [==============================] - 20s 321ms/step - accuracy: 0.9462 - loss: 0.4484 - val_loss: 0.2590 - val_accuracy: 0.9444
Epoch 5/20
61/61 [==============================] - 20s 323ms/step - accuracy: 0.9600 - loss: 0.3198 - val_loss: 0.1842 - val_accuracy: 0.9444
Epoch 6/20
61/61 [==============================] - 20s 325ms/step - accuracy: 0.9549 - loss: 0.3837 - val_loss: 1.3335 - val_accuracy: 0.9444
Epoch 7/20
61/61 [==============================] - 20s 321ms/step - accuracy: 0.9544 - loss: 0.3626 - val_loss: 0.8762 - val_accuracy: 0.9444
Epoch 8/20
61/61 [==============================] - 20s 323ms/step - accuracy: 0.9600 - loss: 0.3469 - val_loss: 0.2045 - val_accuracy: 0.9444
Epoch 9/20
61/61 [==============================] - 20s 336ms/step - accuracy: 0.9672 - loss: 0.2943 - val_loss: 5.9072e-06 - val_accuracy: 1.0000
Epoch 10/20
61/61 [==============================] - 20s 334ms/step - accuracy: 0.9723 - loss: 0.2769 - val_loss: 4.0530e-06 - val_accuracy: 1.0000
Epoch 11/20
61/61 [==============================] - 21s 339ms/step - accuracy: 0.9713 - loss: 0.2478 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 12/20
61/61 [==============================] - 21s 344ms/step - accuracy: 0.9851 - loss: 0.1030 - val_loss: 6.6227e-09 - val_accuracy: 1.0000
Epoch 13/20
61/61 [==============================] - 20s 335ms/step - accuracy: 0.9841 - loss: 0.1818 - val_loss: 0.0014 - val_accuracy: 1.0000
Epoch 14/20
61/61 [==============================] - 21s 342ms/step - accuracy: 0.9851 - loss: 0.1116 - val_loss: 3.1127e-07 - val_accuracy: 1.0000
Epoch 15/20
61/61 [==============================] - 21s 346ms/step - accuracy: 0.9836 - loss: 0.2131 - val_loss: 0.1223 - val_accuracy: 0.9444
Epoch 16/20
61/61 [==============================] - 21s 338ms/step - accuracy: 0.9862 - loss: 0.1407 - val_loss: 1.1855e-06 - val_accuracy: 1.0000
Epoch 17/20
61/61 [==============================] - 21s 345ms/step - accuracy: 0.9790 - loss: 0.1318 - val_loss: 1.3245e-08 - val_accuracy: 1.0000
Epoch 18/20
61/61 [==============================] - 21s 345ms/step - accuracy: 0.9795 - loss: 0.1567 - val_loss: 8.4108e-07 - val_accuracy: 1.0000
Epoch 19/20
61/61 [==============================] - 21s 343ms/step - accuracy: 0.9831 - loss: 0.1475 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 20/20
61/61 [==============================] - 21s 352ms/step - accuracy: 0.9810 - loss: 0.2066 - val_loss: 0.0000e+00 - val_accuracy: 1.0000

inceptionv3 gave accuracy 0.94.

'''

'''
Error's come accrossed:

1. ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.
solution: pip install --upgrade keras numpy pandas sklearn pillow (https://github.com/asataniAIR/Image_DL_Tutorial/issues/4)

2. Internal: Invoking GPU asm compilation is supported on Cuda non-Windows platforms onlyRelying on driver to perform ptx compilation. Modify $PATH to customize ptxas location.
ref: https://github.com/tensorflow/models/issues/7640

3. could not synchronize on CUDA context: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered :: 0x00007FFE2B93BA05	tensorflow::CurrentStackTrace. GPU sync failed
solution: restart the program. (https://stackoverflow.com/questions/51112126/gpu-sync-failed-while-using-tensorflow)
'''