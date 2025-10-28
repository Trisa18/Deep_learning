import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt


#load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Original x_train shape: {x_train.shape}")
print(f"Original y_train shape: {x_train.dtype}")
print(f"Original x_test shape: {x_test.shape}")
print(f"Original y_test shape: {y_test.shape}")
print(x_train[0])
#plt.imshow(x_train[0], cmap='gray')
#plt.show()
print("**********************")
print(f"Label of first image: {y_train[0]}") 

#normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#to_categorical
print(f"Before label is: {y_train[100]}")
y_train = to_categorical(y_train)
print(f"After: label y_train[0]: {y_train[100]}")
y_test = to_categorical(y_test)


#architecture of the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  #flatten the 28x28 images to 784-dimensional vectors
model.add(Dense(128, activation='relu'))  #hidden layer with 128 neurons and ReLU activation
model.add(Dense(10, activation='softmax'))  #output layer with 10 neurons (one for each class) and softmax activation

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#train the model
result=model.fit(x_train, y_train, epochs=5, batch_size=64 , validation_split=0.2)

#evaluate the model
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print(f'Train loss: {train_loss}')
print(f'Train accuracy: {train_accuracy}')


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(result.history.keys())
print(result.history.values())
print(result.history)

#visualize training history
plt.plot(result.history['val_loss'], label='validation loss', color='blue')
plt.plot(result.history['loss'], label='Train loss', color='green')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(result.history['val_accuracy'], label='validation accuracy', color='blue')
plt.plot(result.history['accuracy'], label='Train accuracy', color='green')
plt.title('Training Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()






