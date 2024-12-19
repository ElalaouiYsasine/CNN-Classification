import tf
from keras import Model
from keras.src.applications.vgg19 import VGG19
from keras.src.layers import Flatten, Dense

vgg = VGG19(input_shape=(224,224,3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
# ajout d'une couche de sortie
prediction = Dense(3, activation='softmax')(x)
model3 = Model(inputs=vgg.input, outputs=prediction)
model3.summary()

model3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




feature_extractor = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
