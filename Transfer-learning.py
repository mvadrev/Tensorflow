import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from PIL import Image

base_dir = "images"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_cats = os.path.join(train_dir, "cats")
val_cats = os.path.join(validation_dir, "cats")
train_dogs = os.path.join(train_dir, "dogs")
val_dogs = os.path.join(validation_dir, "dogs")

num_cats_train = len(os.listdir(train_cats))
num_dogs_train = len(os.listdir(train_dogs))
num_cats_val = len(os.listdir(val_cats))
num_dogs_val = len(os.listdir(val_dogs))

total_train = num_cats_train+num_dogs_train
total_val = num_cats_val + num_dogs_val

batch_size = 32
img_shape = 224  # Mobile Net image dimensions

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_image_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(img_shape, img_shape), class_mode="binary")

val_image_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size, directory=validation_dir, shuffle=False, target_size=(img_shape, img_shape), class_mode="binary")

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

mobile_net = hub.KerasLayer(URL, input_shape=(img_shape, img_shape, 3))
mobile_net.trainable = False

model = tf.keras.models.Sequential([
    mobile_net,
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

epochs = 1

history = model.fit_generator(
    train_image_gen,
    steps_per_epoch=int(np.ceil(total_train/float(batch_size))),
    epochs=epochs,
    validation_data=val_image_gen,
    validation_steps=int(np.ceil(total_val/float(batch_size)))
)


def prepare(filepath):
    IMG_SIZE = 50

    img = Image.open('./g.jpg')  # image extension *.png,*.jpg
    new_width = 50
    new_height = 50
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    # format may what u want ,*.png,*jpg,*.gif
    img.save('gs.jpg')

    img = plt.imread("./gs.jpg")
    imgplot = plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


predictions = model.predict([prepare('gs.jpg')])
print("The predictions are", predictions)
