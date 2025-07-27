import cv2
import math
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Canvases
draw_canvas = np.zeros((480, 640), dtype=np.uint8)

# Drawing points
points = []
drawing = False
was_drawing = False

# Final expression string
expression = ""
last_pred_char = ""

# --- Gesture Detection Functions ---
def is_index_finger_up(landmarks):
    return landmarks[8].y < landmarks[5].y

def is_fist(landmarks):
    distances = [math.hypot(landmarks[i].x - landmarks[i+2].x, landmarks[i].y - landmarks[i+2].y) for i in range(4)]
    return all(dist < 0.1 for dist in distances)

def is_palm_open(landmarks):
    return landmarks[20].y < landmarks[4].y

def is_peace_sign(landmarks):
    return (
        landmarks[8].y < landmarks[6].y and   # Index finger up
        landmarks[12].y < landmarks[10].y and # Middle finger up
        landmarks[16].y > landmarks[14].y and # Ring finger down
        landmarks[20].y > landmarks[18].y     # Pinky finger down
    )

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark  # ✅ Define landmarks properly here

            drawing = is_index_finger_up(landmarks)

            # Drawing logic
            if drawing:
                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)
                points.append((x, y))
                for i in range(1, len(points)):
                    cv2.line(draw_canvas, points[i-1], points[i], 255, 4)

            if was_drawing and not drawing and len(points) > 10:
                cv2.imwrite("symbol.png", draw_canvas.copy())
                print("Symbol captured and saved as 'symbol.png'")

            was_drawing = drawing

            # === Gesture detection for actions ===
            if is_fist(landmarks):  # Submit symbol
                if len(points) > 10:
                    symbol_img = draw_canvas.copy()
                    symbol_img = cv2.resize(symbol_img, (28, 28))
                    symbol_img = symbol_img.astype("float32") / 255.0
                    symbol_img = np.expand_dims(symbol_img, axis=-1)
                    symbol_img = np.expand_dims(symbol_img, axis=0)
                    prediction = model.predict(symbol_img)
                    pred_class = np.argmax(prediction)
                    pred_char = str(pred_class) if pred_class <= 9 else ['+', '-', '*', '/'][pred_class - 10]
                    expression += pred_char
                    last_pred_char = pred_char  # ✅ Save it to show on screen
                    print(f"Added: {pred_char}")
                draw_canvas[:] = 0
                points.clear()

            elif is_palm_open(landmarks):  # Evaluate
                # Blinking "Evaluating..."
                for _ in range(3):
                    draw_canvas[:] = 0
                    cv2.putText(draw_canvas, "Evaluating...", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
                    cv2.imshow("Draw", draw_canvas)
                    cv2.waitKey(300)
                    draw_canvas[:] = 0
                    cv2.imshow("Draw", draw_canvas)
                    cv2.waitKey(300)

                # Now evaluate
                try:
                    result = str(eval(expression))
                    expression += "=" + result
                    print(f"Evaluated: {result}")
                    cv2.putText(draw_canvas, f"Result: {result}", (10, 80),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, 255, 2)
                    cv2.waitKey(2000)  # Wait for 2 seconds
                    draw_canvas[:] = 0
                    points.clear()
                    expression = ""

                except Exception as e:
                    expression += "=ERR"
                    print(f"Error evaluating expression: {e}")
                    cv2.putText(draw_canvas, "Eval Error", (10, 80),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, 255, 2)

            elif is_peace_sign(landmarks):  # Clear canvas
                draw_canvas[:] = 0
                points.clear()
                expression = ""
                print("Canvas cleared with peace sign")

    # Display current expression
    cv2.putText(draw_canvas, "Expr: " + expression, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
    cv2.putText(draw_canvas, f"Last: {last_pred_char}", (10, 420),
            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    # Show outputs
    cv2.imshow("Webcam", frame)
    cv2.imshow("Draw", draw_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
