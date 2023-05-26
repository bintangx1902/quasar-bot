# import cv2
# import tensorflow as tf
# import keras
# import numpy as np
#
# # Load your TensorFlow machine learning model
# model = tf.keras.models.load_model(r'C:\Users\binta\Desktop\raw\model\SAR-terrain-v1.5.h5')
#
# # Open a video capture object
# cap = cv2.VideoCapture('./dataset/train2.mp4')  # 0 for default camera, or provide the path to a video file
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set camera resolution to 1280x720
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# labels = ['RETAK', 'TANGGA']
#
#
# def postprocess_outputs(predictions):
#     box_x, box_y, box_w, box_h, class_index = None, None, None, None, np.argmax(predictions)
#     return box_x, box_y, box_w, box_h, class_index
#
#
# def draw_box(frame, box_x, box_y, box_w, box_h, class_label):
#     f_copy = frame.copy()
#     if box_x is not None and box_y is not None and box_w is not None and box_h is not None:
#         cv2.rectangle(f_copy, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
#         if class_label is not None:
#             cv2.putText(f_copy, class_label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     return f_copy
#
#
# # Loop over video frames
# while True:
#     ret, frame = cap.read()
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_resized = cv2.resize(frame_rgb, (192, 255))
#     frame_normalized = frame_resized / 255.0
#     frame_expanded = tf.expand_dims(frame_normalized, axis=0)
#     predictions = model.predict(frame_expanded)
#     print(predictions)
#     box_x, box_y, box_w, box_h, class_index = postprocess_outputs(predictions)
#
#     class_label = None
#     if class_index is not None:
#         class_label = labels[class_index]
#
#     # Draw bounding box and class label on video frame
#     frame_with_box = draw_box(frame, box_x, box_y, box_w, box_h, class_label)
#
#     cv2.imshow('Video', frame_with_box)
#
#     if class_label is not None:
#         print(class_label)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# import serial
#
# ser = serial.Serial('/dev/ttyACM0', 9600)
# ser.write(b'some_command')

import cv2
import time
# import RPi.GPIO as GPIO
import numpy as np

# Set GPIO mode
# GPIO.setmode(GPIO.BCM)

# Set motor pins
# motor_pin1 = 18  # Motor Pin 1
# motor_pin2 = 23  # Motor Pin 2

# Set GPIO motor pins as output
# GPIO.setup(motor_pin1, GPIO.OUT)
# GPIO.setup(motor_pin2, GPIO.OUT)

# Set color threshold for orange (adjust the values as needed)
orange_lower = (0, 50, 50)
orange_upper = (20, 240, 240)

# Set camera properties
camera_width = 640
camera_height = 480

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)


# Define motor control functions
def rotate_clockwise():
    # GPIO.output(motor_pin1, GPIO.HIGH)
    # GPIO.output(motor_pin2, GPIO.LOW)
    print('kanan')


def rotate_counter_clockwise():
    # GPIO.output(motor_pin1, GPIO.LOW)
    # GPIO.output(motor_pin2, GPIO.HIGH)
    print('kiri')


def stop_rotation():
    # GPIO.output(motor_pin1, GPIO.LOW)
    # GPIO.output(motor_pin2, GPIO.LOW)
    pass


def center():
    print('center')


# Main loop
while True:
    ret, frame = cap.read()
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame, orange_lower, orange_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if not M['m10'] or not M['m00']:
            M = 0
        else:
            M = M['m10'] / M['m00']
        center_x = int(M)
        if 180 <= center_x <= 330:
            center()
        elif center_x < camera_width // 2:
            rotate_counter_clockwise()
        else:
            rotate_clockwise()
    else:
        stop_rotation()

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # time.sleep(1)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
