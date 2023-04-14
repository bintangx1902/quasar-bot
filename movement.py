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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
