#%%
#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
import numpy as np
import matplotlib.pyplot as plt
import os,datetime
# %%
#2. Data loading  
BATCH_SIZE = 32
IMG_SIZE = (224,224)
data = keras.utils.image_dataset_from_directory('dataset',batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)
#%%
train_size = int(len(data)*.7)
val_size = int(len(data)*.2) # for tuning parameters like number of hidden layers
test_size = int(len(data)*.1) # solely for getting the performance of the model
#%%
train_dataset = data.take(train_size)
val_dataset = data.skip(train_size).take(val_size)
test_dataset = data.skip(train_size+val_size).take(test_size)
# %%
#3. Convert the datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
#4. Create the data augmentation model
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))
# %%
#5. Create the input preprocessing layer
preprocess_input = applications.mobilenet_v2.preprocess_input
# %%
#6. Apply transfer learning
class_names = ['Negative', 'Positive']
nClass = len(class_names)
#(A) Apply transfer learning to create the feature extractor
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV3Large(input_shape=IMG_SHAPE,include_top=False,weights="imagenet",include_preprocessing=False)
base_model.trainable = False
# %%
#(B) Create the classifier
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nClass,activation='softmax')
# %%
#7. Link the layers together to form the model pipeline using functional API
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)
# x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()
# %%
#8. Compile the model
cos_decay = optimizers.schedules.CosineDecay(0.0005,50)
optimizer = optimizers.Adam(learning_rate=cos_decay)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#9. Evaluate the model before training
loss0,acc0 = model.evaluate(pf_test)
print("----------------Evaluation Before Training-------------------")
print("Loss = ",loss0)
print("Accuracy = ",acc0)
# %%
#10. Create tensorboard
base_log_path = r"tensorboard_logs\concrete_cracks_classification"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#11. Model training
EPOCHS = 5
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])
# %%
#12. Follow-up training
base_model.trainable = True
for layer in base_model.layers[:200]:
    layer.trainable = False
base_model.summary()
# %%
#13. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#14. Continue training the model
fine_tune_epoch = 5
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch = history.epoch[-1],callbacks=[tb])
# %%
#15. Evaluate the model after training
test_loss, test_acc = model.evaluate(pf_test)
print("----------------Evaluation After Training---------------")
print("Test loss = ",test_loss)
print("Test accuracy = ",test_acc)
# %%
#16. Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)
#Stack the label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))
#%%
PATH1 = r"C:\Users\USER\Desktop\deep_learning_computer_vision\image_classification\concrete_cracks_classification\saved_models"
print(PATH1)
# %%
#Model save path
model_save_path = os.path.join(PATH1,"concrete_cracks_classification_model.h5")
keras.models.save_model(model,model_save_path)
#%%
#Check if the model can be loaded
model_loaded = keras.models.load_model(model_save_path)
# %%
