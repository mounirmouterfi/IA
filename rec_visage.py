import cv2
import os
import numpy as np


net = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def get_embedding(image):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("Aucun visage détecté dans l'image.")
        return None


    (x, y, w, h) = faces[0]


    roi = image[y:y+h, x:x+w]


    roi_resized = cv2.resize(roi, (96, 96))


    blob = cv2.dnn.blobFromImage(roi_resized, scalefactor=1.0/255, size=(96, 96), mean=(0, 0, 0), swapRB=True)


    net.setInput(blob)


    vec = net.forward()

    return vec.flatten() 

def load_embeddings_from_folder(folder):
    embeddings = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)
            embedding = get_embedding(image)
            if embedding is not None:
                embeddings.append((folder, embedding))  
    return embeddings

def recognize_student(embeddings, unknown_embedding):
    distances = []


    for name, known_embedding in embeddings:
        distance = np.linalg.norm(unknown_embedding - known_embedding)
        distances.append((name, distance))


    distances.sort(key=lambda x: x[1])


    closest_student = distances[0][0]

    return closest_student


mounir_embeddings = load_embeddings_from_folder("Mounir")
adam_embeddings = load_embeddings_from_folder("Adam")


embeddings = mounir_embeddings + adam_embeddings


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  
    unknown_embedding = get_embedding(frame)

    if unknown_embedding is not None:

        student = recognize_student(embeddings, unknown_embedding)

  
        cv2.putText(frame, student, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow('Identified Student', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
