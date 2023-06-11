import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical

class DogBreedClassifier:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.class_dict = {}
        self.images = []
        self.labels = []
        self.model = None
        self.num_classes = 0

    def load_images(self):
        class_index = 0
        for folder_name in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, folder_name)):
                class_name = folder_name.split('-')[1]
                if class_name not in self.class_dict:
                    self.class_dict[class_name] = class_index
                    class_index += 1

                folder_path = os.path.join(self.data_dir, folder_name)
                if len(os.listdir(folder_path)) > 0:
                    for image_name in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_name)
                        try:
                            image = cv2.imread(image_path)
                            image = cv2.resize(image, (64, 64))
                            self.images.append(image)
                            self.labels.append(self.class_dict[class_name])
                        except Exception as e:
                            print(f"Error loading image {image_path}: {str(e)}")

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print("Number of images:",len(self.images))
        print("Number of labels:",len(self.labels))

    def split_data(self):
        train_images, test_images, train_labels, test_labels = train_test_split(
            self.images, self.labels, test_size=0.2, random_state=42
        )
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )

        self.num_classes = len(np.unique(train_labels))
        print("Number of classes:",self.num_classes)
        train_labels = to_categorical(train_labels, self.num_classes)
        val_labels = to_categorical(val_labels, self.num_classes)
        test_labels = to_categorical(test_labels, self.num_classes)

        self.train_images = train_images / 255.0
        self.val_images = val_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def train_model(self):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(self.train_images)

        opt = Adam(learning_rate=0.0001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit(
            datagen.flow(self.train_images, self.train_labels, batch_size=128),
            epochs=100,
            validation_data=(self.val_images, self.val_labels)
        )

    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(
            self.test_images, self.test_labels
        )
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    def predict_breed(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)

        for breed, index in self.class_dict.items():
            if index == predicted_class:
                return breed

        return "Unknown"
