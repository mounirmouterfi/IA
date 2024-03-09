import cv2

# Charger le fichier xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# ici pour Capturer le flux vidéo de la caméra
cap = cv2.VideoCapture(0)  

while True:
   
    ret, frame = cap.read()
    if not ret:
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Pour chaque visage détecté, dessiner un rectangle autour de celui-ci
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
      
  
    cv2.imshow('Face Detection', frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
