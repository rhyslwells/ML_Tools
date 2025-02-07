import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Sample data: 10 samples, 5 features each
X_train = np.random.rand(10, 5)
# Sample labels: 10 samples, 3 classes (one-hot encoded)
y_train = to_categorical(np.random.randint(3, size=(10, 1)), num_classes=3)

# Define a simple model
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model with categorical cross entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2)