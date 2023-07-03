import dogs_classification
import os
import VGG16

data_dir = os.path.join(os.getcwd(), 'Dataset', 'images', 'Images')

image_path = os.path.join(os.getcwd(), 'tes', 'n02086079_146.jpg')

classifier = dogs_classification.DogBreedClassifier(data_dir)
classifier.load_images()
classifier.split_data()
classifier.build_model()
classifier.train_model()
classifier.evaluate_model()

breed = classifier.predict_breed(image_path)
print("Predicted Breed:", breed)

vgg16Classifier = VGG16.DogBreedClassifier(data_dir)
vgg16Classifier.load_images()
vgg16Classifier.split_data()
vgg16Classifier.build_model()
vgg16Classifier.train_model()
vgg16Classifier.evaluate_model()

vgg16Breed = vgg16Classifier.predict_breed(image_path)
print("VGG16 Predicted Breed:", vgg16Breed)