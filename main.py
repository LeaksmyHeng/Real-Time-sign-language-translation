"""
Leaksmy Heng
CS5330
4/9/2025
Final Project
"""
from datetime import datetime

import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image

from deep_learning_framework.constants import Constants
from deep_learning_framework.model import ResNet50
from deep_learning_framework.data_pytorch import index_to_class

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def execute(network_path):
    """
    Main function to integrate the camera with the model to implement real-time sign language translation.

    https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#
    """

    # create a video capture object
    cap = cv2.VideoCapture(0)

    # load the model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = ResNet50().to(device)
    network.load_state_dict(torch.load(network_path))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        is_save = False
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

                    # Preprocess ROI
                    try:
                        roi_to_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(roi_to_rgb)
                        input_tensor = transform(pil_img).unsqueeze(0).to('cuda')
                        # make prediction
                        with torch.no_grad():
                            output = network(input_tensor)
                            ## use this if to get the index and label
                            # pred_index = output.argmax().item()
                            # pred_label = index_to_class[pred_index]

                            # use softmax to get confidential so the hand is not shown if the confident is less then the set threshold
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            max_prob, pred_index  = torch.max(probabilities, 1)

                            if max_prob.item() >= Constants.CONFIDENTIAL_THRESHOLD:
                                pred_label = index_to_class[pred_index.item()]
                                color = (36, 255, 12)
                            else:
                                pred_label = "Low Confidence"
                                color = (128, 128, 128)

                        cv2.putText(image, pred_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    except cv2.error:
                        print("Invalid ROI")
                        continue

                    # Save/display extracted region
                    # cv2.imshow("Extracted Hand", hand_roi)
                    if is_save:
                        cv2.imwrite(f'hand_crop_{datetime.now().timestamp()}.jpg', hand_roi)
                        is_save = False

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Hands', cv2.flip(image, 1))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                is_save = True

    cap.release()


if __name__ == '__main__':
    print('Running integration.')
    execute(network_path=r'.\deep_learning_framework\output\train_from_scratch\model.pth')
    print('Bye bye!')
