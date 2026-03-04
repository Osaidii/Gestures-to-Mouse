import math
import time
import mediapipe as mp
import cv2
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap =  cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

points_ids = [0, 5, 9, 13, 17]

dead_zone = 20

PINCH_START = 15

def main():
    prev_x, prev_y = screen_width / 2, screen_height / 2
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while True:
            attempt = 0
            success, img = cap.read()
            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1
                print("here")
            if not success:
                print("Could not read camera image")
                break
            img = cv2.flip(img, 1)
            h = img.shape[0]
            w = img.shape[1]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                    )

                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    middle_tip = hand_landmarks.landmark[9]

                    cx = sum(hand_landmarks.landmark[id].x for id in points_ids) / len(points_ids)
                    cy = sum(hand_landmarks.landmark[id].y for id in points_ids) / len(points_ids)

                    cv2.circle(img, (int(cx * w), int(cy * h)), 5, (0, 255, 0), -5)

                    screen_x = cx * screen_width
                    screen_y = cy * screen_height

                    dx = screen_x - prev_x
                    dy = screen_y - prev_y

                    if math.hypot(dx, dy) > dead_zone:
                        pyautogui.moveTo(
                            int(cx * screen_x),
                            int(cy * screen_y),
                            duration=0
                        )
                        prev_x = screen_x
                        prev_y = screen_y

                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    mx, my = int(middle_tip.x * w), int(middle_tip.y * h)

                    pinch_distance = math.hypot(ix - tx, iy - ty)
                    left_pinch_distance = math.hypot(ix - mx, iy - my)

                    if left_pinch_distance < PINCH_START:
                        pyautogui.leftClick()

                    if pinch_distance < PINCH_START:
                        pyautogui.click()

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





    cap.release()
    cv2.destroyAllWindows()

main()