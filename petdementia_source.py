import keras
from keras.engine import Model
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.layers import Flatten, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout

'''
MODEL SETUP
'''
weights_to_use = 'imagenet'

base_model = InceptionV3(weights=weights_to_use,
                 include_top=False,
                 input_shape=(input_size,input_size,3))
model_name_str = 'InceptionV3'

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = True
the_optimizer=keras.optimizers.Adam(lr=0.0001)

loss = 'categorical_crossentropy'

model.compile(optimizer=the_optimizer, 
              loss=loss, 
              metrics=['accuracy'])

'''
TRAIN DATA GENERATOR
'''
train_datagen = ImageDataGenerator(
    rotation_range=8,
    shear_range = np.pi / 16,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.08,
    horizontal_flip=False,
    vertical_flip=False)


class_mode = 'categorical'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_size,input_size),
        batch_size=batch_size,
        class_mode=class_mode,
        classes = classes_to_use,
        shuffle = True
)

'''
VAL DATA GENERATOR
'''
val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(input_size,input_size),
        batch_size=batch_size,
        class_mode=class_mode,
        classes = classes_to_use,
        shuffle = False
)

'''
MODEL TRAINING
'''
model.fit_generator(train_generator,
                    steps_per_epoch    = n_train_samples // batch_size, 
                    epochs             = nb_epoch, 
                    verbose            = True,
                    validation_data    = val_generator,
                    validation_steps   = n_val_samples // batch_size,
                    class_weight       = 'auto',
                    initial_epoch      = initial_epoch)
