import cv2
from fer import FER

# Charger le modèle FerPlus
detector = FER()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = detector.detect_emotions(frame)
    for result in results:
        if "happy" in result["emotions"]:
            happy_score = result["emotions"]["happy"] * 100
            cv2.putText(frame, f"happy: {happy_score:.0f}%", (result["box"][0], result["box"][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (result["box"][0], result["box"][1]), (result["box"][0] + result["box"][2], result["box"][1] + result["box"][3]), (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
