import cv2
import numpy as np
from machine import *
from movement import *
import time
import tensorflow as tf

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'./dataset/train2.mp4')

if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

thresh = .6

model = tf.keras.models.load_model(r'C:\Users\binta\Desktop\raw\model\SAR-terrain-v1.5.h5')
labels = ['RETAK', 'TANGGA']

while True:
    # _, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)
    # col_middle = thresh[:, thresh.shape[1] // 2]

    binary, frame = get_binary_image(cap, cv2)
    if binary is None:
        break

    conv = frame_conv_opt(binary, 2, 3)
    # print(conv)

    """ machine learning """
    frame_resized = cv2.resize(frame, (192, 255))
    frame_normalized = frame_resized / 255.0
    frame_expanded = tf.expand_dims(frame_normalized, axis=0)
    predictions = model.predict(frame_expanded)

    box_x, box_y, box_w, box_h, class_index = postprocess_outputs(predictions, np)

    class_label = None
    if class_index is not None:
        class_label = labels[class_index]

    frame_with_box = draw_box(frame, box_x, box_y, box_w, box_h, class_label, cv2)

    if ((conv[1, :2] < thresh).all() and conv[0, :1] < thresh) or (conv[0, :1] < thresh and conv[1, :1] < thresh):
        """ move left will be ignored if the condition of angled left acceptable """
        if ((conv[0, :2] < thresh).all() and (conv[1, :] < thresh).all()) or \
                (conv[0, :1] < thresh and (conv[1, :] < thresh).all):
            move_angled_left()
        else:
            move_left()
    elif ((conv[0, :] < thresh).all() and (conv[1, :] < thresh).all()) or \
            ((conv[1, :]).all() and np.array_equal(conv[0, :], [False, True, False])) or \
            (conv[1, :] < thresh).all() or ((conv[0, :] < thresh)[1] and (conv[1, :] < thresh)[1]) or \
            (conv[1, :] < thresh)[1]:
        move_forward()

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

    cv2.imshow('Video', frame_with_box)

    if class_label is not None:
        print(class_label)

    cv2.imshow("Binary Footage", binary)

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('c'):
        break

stop()
