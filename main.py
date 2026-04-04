import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        texto = "No se detecta mano"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Coordenadas de algunos dedos
                landmarks = hand_landmarks.landmark

                # Comparacion simple: dedo indice levantado o no
                indice_tip = landmarks[8]
                indice_pip = landmarks[6]

                medio_tip = landmarks[12]
                medio_pip = landmarks[10]

                if indice_tip.y < indice_pip.y and medio_tip.y < medio_pip.y:
                    texto = "Gesto: Paz / 2 dedos"
                elif indice_tip.y < indice_pip.y:
                    texto = "Gesto: 1 dedo"
                else:
                    texto = "Mano detectada"

        cv2.putText(frame, texto, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar imagen fake segun gesto
        if texto == "Gesto: Paz / 2 dedos":
            cv2.putText(frame, "Mostrar imagen PAZ", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif texto == "Gesto: 1 dedo":
            cv2.putText(frame, "Mostrar imagen 1 DEDO", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Detector de gestos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()