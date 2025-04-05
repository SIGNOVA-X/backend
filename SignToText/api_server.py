from flask import Flask, request, jsonify
import numpy as np
import math
import cv2
import os
import werkzeug
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Then you can use logger throughout your code, for example:
logger.info('Hand detection started')
app = Flask(__name__)

# Initialize hand detector and model
hd = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.5)
hd2 = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.5)
model = load_model('cnn8grps_rad1_model.h5')
offset = 29

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def predict_gesture(test_image, pts):
    white = test_image.reshape(1, 400, 400, 3)
    prob = np.array(model.predict(white)[0], dtype='float32')
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    ch3 = np.argmax(prob, axis=0)
    prob[ch3] = 0

    pl = [ch1, ch2]

    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]

    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    # con for [l][pqz]
    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    # con for [pqz][yj]
    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    # condition for [yj][x]
    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]
    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    fg = 19
    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
        (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 7

    # con for [uvr]
    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and
                pts[0][0] + fg < pts[20][0]) and not (
                pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    # con for [w]
    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # -------------------------condn for 8 groups  ends

    # -------------------------condn for subgroups  starts
    #
    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = 'N'

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42:
            ch1 = 'C'
        else:
            ch1 = 'O'

    if ch1 == 3:
        if (distance(pts[8], pts[12])) > 72:
            ch1 = 'G'
        else:
            ch1 = 'H'

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42:
            ch1 = 'Y'
        else:
            ch1 = 'J'

    if ch1 == 4:
        ch1 = 'L'

    if ch1 == 6:
        ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]:
                ch1 = 'Z'
            else:
                ch1 = 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'
        if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'R'

    if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1=" "

    if ch1 == 'E' or ch1=='Y' or ch1=='B':
        if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1="next"

    if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
            ch1 = 'Backspace'

    return ch1

def predict_from_file(image_path):
    """
    Predicts the sign language for a given image file.
    """
    try:
        # Read and process the image
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': 'Could not read image file'}
        if frame.size == 0:
            return {'error': 'Empty image file'}
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        cv2image = cv2.flip(frame, 1)

        hands, img = hd.findHands(cv2image, draw=False, flipType=True)

        if not hands:
            return {'prediction': 'No hand detected'}

        hand = hands[0]
        x, y, w, h = hand['bbox']
        image = cv2image[y - offset:y + h + offset, x - offset:x + w + offset]

        # Create white background
        white = np.ones((400, 400, 3), np.uint8) * 255

        handz, img = hd2.findHands(image, draw=False, flipType=True)
        if not handz:
            return {'prediction': 'Hand not properly detected'}

        hand = handz[0]
        pts = hand['lmList']

        offset_x = ((400 - w) // 2) - 15
        offset_y = ((400 - h) // 2) - 15

        # Draw hand landmarks on white image
        for t in range(0, 4, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(5, 8, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(9, 12, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(13, 16, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(17, 20, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        cv2.line(white, (pts[5][0] + offset_x, pts[5][1] + offset_y), (pts[9][0] + offset_x, pts[9][1] + offset_y), (0, 255, 0),
                 3)
        cv2.line(white, (pts[9][0] + offset_x, pts[9][1] + offset_y), (pts[13][0] + offset_x, pts[13][1] + offset_y), (0, 255, 0),
                 3)
        cv2.line(white, (pts[13][0] + offset_x, pts[13][1] + offset_y), (pts[17][0] + offset_x, pts[17][1] + offset_y),
                 (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + offset_x, pts[0][1] + offset_y), (pts[5][0] + offset_x, pts[5][1] + offset_y), (0, 255, 0),
                 3)
        cv2.line(white, (pts[0][0] + offset_x, pts[0][1] + offset_y), (pts[17][0] + offset_x, pts[17][1] + offset_y), (0, 255, 0),
                 3)

        for i in range(21):
            cv2.circle(white, (pts[i][0] + offset_x, pts[i][1] + offset_y), 2, (0, 0, 255), 1)

        # Predict the gesture
        prediction = predict_gesture(white, pts)

        return {'prediction': str(prediction)}

    except Exception as e:
        return {'error': str(e)}

def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filename = werkzeug.utils.secure_filename(file.filename)
    file.save(filename)
    
    try:
        # Read and process the image
        frame = cv2.imread(filename)
        if frame is None:
           return jsonify({'error': 'Could not read image file'}), 400
        if frame.size == 0:
           return jsonify({'error': 'Empty image file'}), 400
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        cv2image = cv2.flip(frame, 1)
        
        hands, img = hd.findHands(cv2image, draw=False, flipType=True)
        
        if not hands:
            return jsonify({'prediction': 'No hand detected'}), 200
            
        hand = hands[0]
        x, y, w, h = hand['bbox']
        image = cv2image[y - offset:y + h + offset, x - offset:x + w + offset]
        
        # Create white background
        white = np.ones((400, 400, 3), np.uint8) * 255
        
        handz, img = hd2.findHands(image, draw=False, flipType=True)
        if not handz:
            return jsonify({'prediction': 'Hand not properly detected'}), 200
            
        hand = handz[0]
        pts = hand['lmList']
        
        offset_x = ((400 - w) // 2) - 15
        offset_y = ((400 - h) // 2) - 15

        # Draw hand landmarks on white image
        for t in range(0, 4, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(5, 8, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(9, 12, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(13, 16, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        for t in range(17, 20, 1):
            cv2.line(white, (pts[t][0] + offset_x, pts[t][1] + offset_y), (pts[t + 1][0] + offset_x, pts[t + 1][1] + offset_y),
                     (0, 255, 0), 3)
        cv2.line(white, (pts[5][0] + offset_x, pts[5][1] + offset_y), (pts[9][0] + offset_x, pts[9][1] + offset_y), (0, 255, 0),
                 3)
        cv2.line(white, (pts[9][0] + offset_x, pts[9][1] + offset_y), (pts[13][0] + offset_x, pts[13][1] + offset_y), (0, 255, 0),
                 3)
        cv2.line(white, (pts[13][0] + offset_x, pts[13][1] + offset_y), (pts[17][0] + offset_x, pts[17][1] + offset_y),
                 (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + offset_x, pts[0][1] + offset_y), (pts[5][0] + offset_x, pts[5][1] + offset_y), (0, 255, 0),
                 3)
        cv2.line(white, (pts[0][0] + offset_x, pts[0][1] + offset_y), (pts[17][0] + offset_x, pts[17][1] + offset_y), (0, 255, 0),
                 3)

        for i in range(21):
            cv2.circle(white, (pts[i][0] + offset_x, pts[i][1] + offset_y), 2, (0, 0, 255), 1)
    
        # Predict the gesture
        prediction = predict_gesture(white, pts)
        
        return jsonify({'prediction': str(prediction)}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)