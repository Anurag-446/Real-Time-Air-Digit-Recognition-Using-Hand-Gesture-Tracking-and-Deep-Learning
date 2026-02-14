import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("digit_model.keras")

# Mediapipe Hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Previous finger position
prev_x, prev_y = None, None

predicted_digit = "None"


# Function to check finger up/down
def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    fingers = []

    # Thumb (simple check)
    if hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (compare y)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # [thumb, index, middle, ring, pinky]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_mode = False  # default: stop writing

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_state = fingers_up(hand_landmarks)

            # DRAW MODE: Index up + Middle down (pointing gesture)
            if finger_state[1] == 1 and finger_state[2] == 0:
                draw_mode = True
            else:
                draw_mode = False

            # Index finger tip position
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Show fingertip
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # Draw only if draw_mode is ON
            if draw_mode:
                cv2.putText(frame, "Mode: Writing", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 12)

                prev_x, prev_y = x, y
            else:
                cv2.putText(frame, "Mode: Stop", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None

    # Show prediction
    cv2.putText(frame, f"Predicted: {predicted_digit}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(frame, "C=Clear | P=Predict | Q=Quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Combine frame and canvas
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("Hand Digit Recognition", combined)

    key = cv2.waitKey(1) & 0xFF

    # Clear canvas
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        predicted_digit = "None"

    # Predict digit(s)
    if key == ord('p'):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Make drawing strong
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            predicted_digit = "No Drawing!"
        else:
            # Sort contours left-to-right
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            result_digits = []

            for cnt in contours:
                x1, y1, w_box, h_box = cv2.boundingRect(cnt)

                # Ignore noise
                if w_box < 15 or h_box < 15:
                    continue

                # Crop digit
                digit_crop = thresh[y1:y1 + h_box, x1:x1 + w_box]

                # Add padding around digit
                pad = 20
                digit_crop = cv2.copyMakeBorder(
                    digit_crop, pad, pad, pad, pad,
                    cv2.BORDER_CONSTANT, value=0
                )

                # Make it square (MNIST style)
                h2, w2 = digit_crop.shape
                if h2 > w2:
                    diff = h2 - w2
                    digit_crop = cv2.copyMakeBorder(
                        digit_crop, 0, 0, diff // 2, diff - diff // 2,
                        cv2.BORDER_CONSTANT, value=0
                    )
                elif w2 > h2:
                    diff = w2 - h2
                    digit_crop = cv2.copyMakeBorder(
                        digit_crop, diff // 2, diff - diff // 2, 0, 0,
                        cv2.BORDER_CONSTANT, value=0
                    )

                # Resize to 28x28
                digit_28 = cv2.resize(digit_crop, (28, 28))

                # Normalize
                digit_28 = digit_28 / 255.0
                digit_28 = digit_28.reshape(1, 28, 28, 1)

                pred = model.predict(digit_28, verbose=0)
                result_digits.append(str(np.argmax(pred)))

            if len(result_digits) == 0:
                predicted_digit = "No Digit Found!"
            else:
                predicted_digit = "".join(result_digits)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
