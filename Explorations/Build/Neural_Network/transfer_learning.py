import tensorflow_datasets as tfds
import tensorflow as tf
dataset, info = tfds.load("tf_flowers",
                          as_supervised=True,
                          with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info. features["label"].num_classes
test_set_raw, valid_set_raw, train_set_raw
            = tfds.load("tf_flowers",
                        split=["train[:10%]",
                               "train[10%:25%]",
                               "train[25%:]"],
                        as_supervised=True)
                        
def preprocess(image,label):
    resized_image = tf.image.resize(image, [224,224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


batch_size = 32
train_set = train_set_raw.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

base_model = tf.keras.applications.xception.Xception(weights="imagenet",
                                            include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

  
#freezing previous layers
for layer in base_model.layers:
  layer.trainable = False

optimizer = tf.keras.optimizers.SGD(lr=0.2,
                                    momentum=0.9,
                                    decay=0.01)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])


history = model.fit(train_set,
                    epochs=5,
                    validation_data=valid_set)

 
for layer in base_model.layers:
  layer.trainable = True


# change runtime to gpu here.
optimizer = tf.keras.optimizers.SGD(lr=0.01,
                                    momentum=0.9,
                                    decay=0.001)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

  
history = model.fit(train_set,
                    epochs=10,
                    validation_data=valid_set)