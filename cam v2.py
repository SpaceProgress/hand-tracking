import cv2
import mediapipe as mp
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    p_time = 0
    fingertips = {4, 8, 12, 16, 20}

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

                    for lm_id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        if lm_id in fingertips:
                            cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)

            c_time = time.time()
            time_diff = c_time - p_time
            fps = 1 / time_diff if time_diff > 0 else 0
            p_time = c_time

            cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.imshow("Hand Tracking", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()