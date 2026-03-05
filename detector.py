import math
import time
import mediapipe as mp
import cv2
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
from pynput.keyboard import Controller

keyboard = Controller()

root = tk.Tk()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap =  cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

points_ids = [0, 5, 9, 13, 17]

dead_zone = 15

PINCH_START = 20

def main():
    prev_x, prev_y = screen_width / 2, screen_height / 2
    with mp_hands.Hands(
            max_num_hands=2,
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
            if not success:
                print("Could not read camera image")
                break
            img = cv2.flip(img, 1)
            h = img.shape[0]
            w = img.shape[1]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                )

                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]

                cx = sum(hand_landmarks.landmark[id].x for id in points_ids) / len(points_ids)
                cy = sum(hand_landmarks.landmark[id].y for id in points_ids) / len(points_ids)

                cv2.circle(img, (int(cx * w), int(cy * h)), 5, (0, 255, 0), -5)

                screen_x = cx * screen_width
                screen_y = cy * screen_height

                dx = screen_x - prev_x
                dy = screen_y - prev_y

                if math.hypot(dx, dy) > dead_zone:
                    pyautogui.moveTo(
                        int(screen_x),
                        int(screen_y),
                        duration=0
                    )
                    prev_x = screen_x
                    prev_y = screen_y

                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                rx, ry = int(ring_tip.x * w), int(ring_tip.y * h)
                px, py = int(pinky_tip.x * w), int(pinky_tip.y * h)

                t_to_i_pinch_distance = math.hypot(ix - tx, iy - ty)
                t_to_m_pinch_distance = math.hypot(mx - tx, my - ty)
                t_to_r_pinch_distance = math.hypot(rx - tx, ry - ty)
                t_to_p_pinch_distance = math.hypot(px - tx, py - ty)

                if t_to_i_pinch_distance < PINCH_START:
                    pyautogui.leftClick()

                if t_to_m_pinch_distance < PINCH_START:
                    pyautogui.rightClick()

                if t_to_r_pinch_distance < PINCH_START:
                    pyautogui.scroll(-50)

                if t_to_p_pinch_distance < PINCH_START:
                    pyautogui.scroll(50)

            cv2.imshow("Showing Camera", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    cv2.destroyAllWindows()

if  __name__ == "__main__":
    main()