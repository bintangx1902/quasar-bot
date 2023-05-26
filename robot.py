import cv2
import numpy as np
from machine import *
from movement import *
import time
import tensorflow as tf
from tensorflow import keras
import keras

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'./dataset/train2.mp4')

if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

thresh = .4

terrain_model_path = r'C:\a-dev\competition\model\model-baik.h5'
human_model_path = r'C:\a-dev\competition\model\model-orang.h5'
terrain_model = keras.models.load_model(terrain_model_path)
human_model = keras.models.load_model(human_model_path)

terrain_labels = ['RETAKAN', 'BAGUS', 'ANAK_TANGGA']
human_labels = ['BAGUS', 'HUMAN']

rotation_servo_pos = 0
is_pinched = False

orange_lower = (0, 50, 50)
orange_upper = (20, 240, 240)

while True:
    # _, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)
    # col_middle = thresh[:, thresh.shape[1] // 2]

    binary, frame = get_binary_image(cap, cv2)
    if binary is None:
        break

    contours = find_contours(frame, cv2, orange_lower, orange_upper)
    conv = frame_conv_opt(binary, 2, 3)
    # print(conv)

    """ machine learning """
    frame_resized = cv2.resize(frame, (192, 255))
    frame_normalized = frame_resized / 255.0
    frame_expanded = tf.expand_dims(frame_normalized, axis=0)
    human_predictions = human_model.predict(frame_expanded)
    terrain_predictions = terrain_model.predict(frame_expanded)

    box_x, box_y, box_w, box_h, class_index_terain = postprocess_outputs(terrain_predictions, np)
    *_, class_index_human = postprocess_outputs(human_predictions, np)

    terrain_label, human_label = None, None
    terrain_label = terrain_labels[class_index_terain] if class_index_terain is not None else None
    human_label = human_labels[class_index_human] if class_index_human is not None else None
    # frame_with_box = draw_box(frame, box_x, box_y, box_w, box_h, class_label, cv2)

    if human_label == 'HUMAN':
        """ 
        1. maju sampai rotasi motor 90 atau 270 derajat
        2. apabila sudah sampai maka stop movement
        3. gerakan capit
        4. apabila tidak sampai maka geser ke arah
        5. capit dan naikan
        """

        # do_movement(conv, thresh)
        # rotation_servo_pos = camera_claw_move(rotation_servo_pos)
        camera_claw_move(0, contours, cv2, width)

        # if rotation_servo_pos == 90 or rotation_servo_pos == 270:
        #     """ stop movement """
        #     claw_human()
        #     if is_pinched:
        #         rotation_servo_pos = camera_claw_move(0)
        #         do_movement()


    # if (conv[1, :] < THRESH).all() and (conv[0, :] < THRESH)[1]:
    #     move_forward()
    #
    # elif (conv[1, :2] < .5).all() and (conv[0, :] > .5).all():
    #     if (conv[1, :2] < .5).all() and (conv[0, :1] > .5).all():
    #         move_angled_left()

    # moments = cv2.moments(binary)
    #
    # if moments["m00"] > 0:
    #     cx = int(moments["m10"] / moments["m00"])
    #     cy = int(moments["m01"] / moments["m00"])
    # else:
    #     cx = None
    #     cy = None
    #
    # if cx is not None:
    #     if cx < binary.shape[1] // 3:
    #         move_left()
    #     elif cx > binary.shape[1] * 2 // 3:
    #         move_right()
    #     else:
    #         move_forward()
    # else:
    #     stop()

    # time.sleep(.05)

    line_y = int(height // 2)

    cv2.line(binary, (0, line_y), (width, line_y), (0, 255, 255), 2)

    line1_x = int(width // 3)
    line2_x = int(2 * width // 3)

    cv2.line(binary, (line1_x, 0), (line1_x, height), (0, 255, 0), 2)
    cv2.line(binary, (line2_x, 0), (line2_x, height), (0, 255, 0), 2)

    # cv2.imshow('Video', frame_with_box)
    #
    # if class_label is not None:
    #     print(class_label)

    cv2.imshow("Real Footage", frame)
    cv2.imshow("Binary Footage", binary)

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('c'):
        break

stop()
