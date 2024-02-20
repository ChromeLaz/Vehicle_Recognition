import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Caricamento del classificatore per il rilevamento del veicolo pre-addestrato
vehicle_cascade = cv2.CascadeClassifier('cars.xml')  # Utilizza il percorso relativo al file di script

# Caricamento del modello preaddestrato MobileNetV2
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Funzione per il rilevamento dei veicoli nell'immagine utilizzando il classificatore Haar
def detect_vehicles(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return vehicles

# Funzione per la predizione della presenza di veicoli nell'immagine utilizzando il modello CNN
def predict_vehicle_presence(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizzazione dell'immagine
    prediction = model.predict(img)
    return prediction[0][0]

# Caricamento dell'immagine
image_path = 'machine_test.jpg'  # Inserisci il percorso dell'immagine
img = cv2.imread(image_path)

# Rilevamento dei veicoli nell'immagine
vehicles = detect_vehicles(img)

# Riconoscimento della presenza di veicoli nell'immagine utilizzando il modello CNN
for (x, y, w, h) in vehicles:
    vehicle_img = img[y:y+h, x:x+w]
    prediction = predict_vehicle_presence(vehicle_img)
    if prediction > 0.5:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Visualizzazione dell'immagine con i veicoli rilevati
cv2.imwrite('detected.jpg', img)
cv2.waitKey()
cv2.destroyAllWindows()
