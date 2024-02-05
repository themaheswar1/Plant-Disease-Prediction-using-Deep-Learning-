#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , Dropout
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')


# In[ ]:


# Define paths to training and testing datasets
train_path = '/content/drive/MyDrive/tomato/train'
test_path = '/content/drive/MyDrive/tomato/val'


# In[ ]:


train_gen = ImageDataGenerator(rescale=(1./255),horizontal_flip=True,shear_range=0.2,zoom_range = 0.2)
test_gen = ImageDataGenerator(rescale=(1./255))  #--> (0 to 255) convert to (0 to 1)


# In[ ]:


train = train_gen.flow_from_directory( '/content/drive/MyDrive/tomato/train',
                                      target_size=(120, 120),
                                      class_mode='categorical',
                                      subset='training',
                                      batch_size=9)
test = test_gen.flow_from_directory('/content/drive/MyDrive/tomato/val',
                                    target_size=(120, 120),
                                      class_mode='categorical',
                                      batch_size=9)


# In[ ]:


train.class_indices


# In[ ]:


# CNN model
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D,Activation
from tensorflow.keras.models import Sequential


# In[ ]:


model = Sequential()

# Block 0
model.add(Conv2D(64, (5, 5), strides=1, padding="same", input_shape=(120, 120, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))

# Block 1
model.add(Conv2D(64, (5, 5), strides=1, padding="same"))
model.add(MaxPooling2D((4, 4)))
model.add(BatchNormalization())
model.add(Activation("relu"))

# Block 2
model.add(Conv2D(128, (3, 3), strides=2, padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.1))  # Adjust dropout rate

# Block 3
model.add(Conv2D(256, (7, 7), strides=2, padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))  # Adjust dropout rate

# Block 4
model.add(Conv2D(512, (3, 3), strides=2, padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))  # Adjust dropout rate

# Block 5
model.add(Conv2D(512, (3, 3), strides=2, padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.15))  # Adjust dropout rate

# Global Average Pooling
model.add(GlobalAveragePooling2D())

# Fully connected layers
model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Adjust dropout rate
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))  # Adjust dropout rate

# Output layer
model.add(Dense(9, activation='softmax'))

model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


#performing early stopping to avoid overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 20, verbose = 1, restore_best_weights = True)


# In[ ]:


history = model.fit(train,batch_size=10,validation_data=test,epochs=50)


# In[ ]:


model.save('/content/drive/MyDrive/Colab Notebooks/FinalDraft.h5')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# Evaluate the model on the validation generator
val_results = model.evaluate(test)

# Extract the metrics from the evaluation results
val_loss = val_results[0]
val_accuracy = val_results[1]

print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')


# In[ ]:


# Predict on the validation generator
val_pred = model.predict(test)

# Convert predicted probabilities to class labels
val_pred_classes = np.argmax(val_pred, axis=1)

# Assuming your validation labels are one-hot encoded
val_true_classes = test.classes

# Calculate metrics
accuracy = accuracy_score(val_true_classes, val_pred_classes)
precision = precision_score(val_true_classes, val_pred_classes, average='weighted')
recall = recall_score(val_true_classes, val_pred_classes, average='weighted')
f1 = f1_score(val_true_classes, val_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[ ]:


# Confusion matrix
conf_matrix = confusion_matrix(val_true_classes, val_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:


from sklearn.metrics import  confusion_matrix


# Calculate sensitivity and specificity for each class
sensitivity_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
specificity_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)

for i in range(len(sensitivity_per_class)):
    print(f'Class {i} - Sensitivity (Recall): {sensitivity_per_class[i]}, Specificity: {specificity_per_class[i]}')


# In[ ]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('/content/drive/MyDrive/FinalDraft.h5')

# Load and preprocess the input image
img_path = '/content/drive/MyDrive/tomato/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/cfbac9ed-82d2-4ccd-9a73-c97c0f92b2e2___UF.GRC_YLCV_Lab 02814.JPG'
img = image.load_img(img_path, target_size=(120, 120))

plt.imshow(img)
plt.title('Input Image')
plt.show()

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)
print(predictions)

class_labels = list(train.class_indices.keys())
# Map predictions to class labels
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]

# Show the predicted class label
print(f'Predicted Class: {predicted_class_label}')


# In[ ]:




