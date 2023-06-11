import dogs_classification


data_dir = 'C:\\Users\\Veljko\\Desktop\\v\\ORI-Projekat\\ORI\\Dataset\\images\\Images'

classifier = dogs_classification.DogBreedClassifier(data_dir)
classifier.load_images()
classifier.split_data()
classifier.build_model()
classifier.train_model()
classifier.evaluate_model()

image_path = 'C:\\Users\\Veljko\\Desktop\\v\\ORI-Projekat\\ORI\\tes\\n02086079_146.jpg'
breed = classifier.predict_breed(image_path)
print("Predicted Breed:", breed)