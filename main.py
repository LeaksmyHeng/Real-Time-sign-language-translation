"""
Leaksmy Heng
CS5330
4/9/2025
Final Project
"""

import cv2
import mediapipe as mp

from deep_learning_framework.constants import Constants

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def execute():
    """
    Main function to integrate the camera with the model to implement real-time sign language translation.

    https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#
    """

    # create a video capture object
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract all x/y coordinates
                    x_coords = [lm.x * image.shape[1] for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * image.shape[0] for lm in hand_landmarks.landmark]

                    # Compute bounding box
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Draw rectangle
                    # without buffer, the box is just around my hand and some part of the finger got cut off slightly
                    # so the margin is really small; therefore, add the buffer to make the ROI a little bit bigger
                    x_min = x_min - Constants.BUFFER
                    y_min = y_min - Constants.BUFFER
                    x_max = x_max + Constants.BUFFER
                    y_max = y_max + Constants.BUFFER
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Extract ROI (Region of Interest)
                    hand_roi = image[y_min:y_max, x_min:x_max]

                    # Save/display extracted region
                    # cv2.imshow("Extracted Hand", hand_roi)
                    # cv2.imwrite("hand_crop.jpg", hand_roi)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == '__main__':
    print('Running integration.')
    execute()
    print('Bye bye!')
