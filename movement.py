# import RPi.GPIO as GPIO
import numpy as np


# GPIO.setmode(GPIO.BOARD)
# GPIO.setwarnings(False)
# GPIO.setup(11, GPIO.OUT)
# GPIO.setup(13, GPIO.OUT)
# GPIO.setup(15, GPIO.OUT)
# GPIO.setup(16, GPIO.OUT)


# Fungsi untuk membaca data dari kamera dan mengubahnya menjadi binary image
def read_camera(cap, cv2):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)

    return thresh, frame


def get_binary_image(cap, cv2):
    # Ambil citra dari kamera
    ret, frame = cap.read()
    if not ret:
        return None

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary, frame


def move_forward():
    # GPIO.output(11, False)
    # GPIO.output(13, True)
    # GPIO.output(15, False)
    # GPIO.output(16, True)
    print('maju')


def move_left():
    # GPIO.output(11, True)
    # GPIO.output(13, False)
    # GPIO.output(15, False)
    # GPIO.output(16, True)
    print('ke kiri')


def move_angled_left():
    print('angled left')


def move_right():
    # GPIO.output(11, False)
    # GPIO.output(13, True)
    # GPIO.output(15, True)
    # GPIO.output(16, False)
    print('ke kanan')


def move_angled_right():
    print('angled right')


def stop():
    # GPIO.output(11, False)
    # GPIO.output(13, False)
    # GPIO.output(15, False)
    # GPIO.output(16, False)
    print('berhenti')


def frame_conv(binary, y_cut, x_cut):
    height, width = binary.shape
    x_width = width // x_cut
    x_height = height // y_cut

    start = 0
    count = 0
    end = x_width

    conv = np.zeros(shape=(y_cut, x_cut))

    """ outer while must be less then height """
    while count != y_cut:
        in_count = 0
        while in_count < x_cut:
            in_end = x_width
            temp = []
            for i in range(start, end):
                for j in range(i, in_end):
                    if i < height and j < width:
                        temp.append(binary[i][j])
                in_end += x_width
            avg = np.average(temp)
            conv[count][in_count] = avg
            in_count += 1
        start += x_height
        count += 1

    return conv


def frame_conv_opt(binary, y_cut, x_cut):
    height, width = binary.shape
    x_width = width // x_cut
    x_height = height // y_cut

    conv = np.zeros(shape=(y_cut, x_cut))

    for count in range(y_cut):
        start = count * x_height
        end = start + x_height
        for in_count in range(x_cut):
            in_end = (in_count + 1) * x_width
            temp = binary[start:end, in_count * x_width:in_end].flatten()
            temp[temp > 0] = 1
            avg = round(np.average(temp), 6)
            conv[count][in_count] = avg

    return conv


def do_movement(conv, thresh):
    if ((conv[1, :2] < thresh).all() and conv[0, :1] < thresh) or (conv[0, :1] < thresh and conv[1, :1] < thresh):
        """ move left will be ignored if the condition of angled left acceptable """
        if ((conv[0, :2] < thresh).all() and (conv[1, :] < thresh).all()) or \
                (conv[0, :1] < thresh and (conv[1, :] < thresh).all):
            move_angled_left()
        else:
            move_left()
    elif (np.array_equal(conv[0, :] < thresh, [False, False, True]) and
          np.array_equal(conv[1, :] < thresh, [False, True, True])) \
            or (np.array_equal(conv[0, :] < thresh, [False, False, True]) or
                np.array_equal(conv[1, :] < thresh, [False, False, True])):
        if ((conv[0, 1:3] < thresh).all() and (conv[1, :] < thresh).all()) or \
                (conv[0, 2:3] < thresh or (conv[1, :] < thresh).all()) or \
                (conv[0, 1:3] < thresh).all() or (conv[1, 1:3]).all():
            move_angled_right()
        else:
            move_right()

    elif ((conv[0, :] < thresh).all() and (conv[1, :] < thresh).all()) or \
            ((conv[1, :]).all() and np.array_equal(conv[0, :], [False, True, False])) or \
            (conv[1, :] < thresh).all() or ((conv[0, :] < thresh)[1] and (conv[1, :] < thresh)[1]) or \
            (conv[1, :] < thresh)[1]:
        move_forward()
    else:
        stop()


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


def find_contours(frame, cv2, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask


def camera_claw_move(current_position, contours, cv2, camera_width):
    pos = current_position
    pos_str = ''
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
            pos_str = 'center'
        elif center_x < camera_width // 2:
            rotate_counter_clockwise()
            pos -= 1
        else:
            pos += 1
            rotate_clockwise()

        if pos >= 90:
            pos = 90
        elif pos <= -90:
            pos = -90

    else:
        stop_rotation()

    return pos, pos_str


def claw_human():
    print('capit')
