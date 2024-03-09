import cv2
import os

def capture_photos(names, num_photos=10):
    for name in names:    
        if not os.path.exists(name):
            os.makedirs(name)
        cap = cv2.VideoCapture(0)
        print(f"Appuyez sur Entrée pour prendre une photo de {name}...")
        count = 0
        while True:
            ret, frame = cap.read()
            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1)
            if key == 13:  # 13 correspond à la touche Entrée
                # Enregistrer l'image avec un nom unique
                count += 1
                photo_name = f"{name}_{count}.jpg"
                cv2.imwrite(os.path.join(name, photo_name), frame)
                print(f"Photo {count} enregistrée comme {photo_name}")
                if count == num_photos:
                    break

        # Libérer la webcam et fermer la fenêtre OpenCV
        cap.release()

    cv2.destroyAllWindows()

# Noms des personnes pour lesquelles vous voulez capturer des photos
names = ["Adam", "Mounir"]

# Utilisation de la fonction
capture_photos(names)
