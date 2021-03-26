# imports initialization
import cv2
import numpy as np
import mediapipe as mp
from playsound import playsound
from datetime import datetime
import math
import imutils
from imutils.video import VideoStream

# get current time
now = datetime.now()
# from object_detector import *
# from skeletal_tracker import skeletalEstimator
medication_taken = False
# webcam initialization
webcamCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcamCapture.set(3, 480)
webcamCapture.set(4, 360)
webcamCapture2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcamCapture2.set(3, 480)
webcamCapture2.set(4, 360)
# sample files for testing
# webcamCapture = cv2.VideoCapture("pillTaking1.mp4")
# coco class names initialization
classNames = []
# confidence threshold value
threshold = 0.6
class_identifier = []
person_coordinates = []
box_coordinates = []
# min and max values for localisation
minX_box_coordinates = []
minY_box_coordinates = []
width_box_coordinates = []
height_box_coordinates = []
fingerKnucklePosition = []
# path locations for object detection
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'
# trained model for medication bottles
configPath_medBottles = 'bottles_label_map.pbtxt'
weightsPath_medBottles = "bottles_model.pb"
# path locations for face detection
faceProtoText = 'deploy.prototxt'
faceCaffeModel = 'res10_300x300_ssd_iter_140000.caffemodel'
# hand media pipe init
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, threshold, threshold)
mpDraw = mp.solutions.drawing_utils
# neural network initializations
# model initialization of neural network for object detection
# net = cv2.dnn.readNetFromTensorflow(weightPath)
net = cv2.dnn_DetectionModel(weightPath, configPath)  # remove after new dataset is implemented
net.setInputSize(480, 360)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# model initialization of neural network for face detection
faceNet = cv2.dnn.readNetFromCaffe(faceProtoText, faceCaffeModel)
# how many times to run through detection loop, change for more loops
repeat_tolerance = 10
detection_tolerance = 5
# output to mp4 file for testing and demonstration purposes
output_filename = 'test_output.mp4'
output_frames_per_second = 20.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file_size = (480, 360)
result = cv2.VideoWriter(output_filename,
                         fourcc,
                         output_frames_per_second,
                         file_size)


# alert sending function
def sendAlert(alert_number):
    if alert_number == 1:
        # test sound
        playsound('sound files/test.mp3')
    if alert_number == 2:
        # "time to take your medication"
        playsound('sound files/time to take your medication.mp3')
    if alert_number == 3:
        # "have you taken your medication?"
        playsound('sound files/have you taken your medication.mp3')
    # if alert_number == x: , allow for additional sounds.


def objectDetector(repeat_tolerance, weights_path_od, config_path_od, model, class_file):
    global classFile
    classFile = class_file
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    print(classNames)  ##### remove this when completed
    # while True:
    for i in range(repeat_tolerance):
        success, img = webcamCapture.read()
        global net
        net = cv2.dnn_DetectionModel(weights_path_od, config_path_od)
        net.setInputSize(480, 360)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        class_ids, confidence_values, bounding_box = net.detect(img, threshold)
        print("Class ID = " + str(
            class_ids)  # + " || ", "Confidence =" + str(confidence_values) ##### remove this when completed
              + "||", "Coordinates " + str(bounding_box))  ##### remove this when completed

        if len(class_ids) != 0:
            for classId, confidence, box in zip(class_ids.flatten(), confidence_values.flatten(), bounding_box):
                cv2.rectangle(img, box, color=(106, 13, 173), thickness=2, lineType=None, shift=None)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 25),
                            cv2.FONT_ITALIC, 1, (106, 13, 173), 2)
                class_identifier.append(classId)
                print(class_identifier)  ##### remove this when completed
                # add the box coordinates to the list.
                if model == "basic":
                    if classId == 44:
                        box_coordinates.append(box)
                        # [ x, y , w , h ]
                        minX_box_coordinates.append(box[0])
                        minY_box_coordinates.append(box[1])
                        width_box_coordinates.append(box[2])
                        height_box_coordinates.append(box[3])

                        print(box_coordinates)  ##### remove this when completed
                # if model == "basic":
                # if classId == 1:
                # person_coordinates.append(box)
                # print(person_coordinates) ##### remove this when completed
        cv2.imshow('TEST', img)
        result.write(img)
        cv2.waitKey(10)


def faceDetection(repeat_tolerance):
    pass


# if minX < localiseX < maxX:
# if minY < localiseY < maxY:
# hand on box
# break
#     else:
#         print("FAIL")
#         if detection_tolerance == 0:
#             sendAlert(2)
#             # recursive loop until medication is taken
#             medDetection(5)
#         else:
#             detection_tolerance = detection_tolerance - 1
#             medDetection(detection_tolerance)


def handDetection(box_coordinates, minX, minY, maxX, maxY):
    # 3 = thumb knuckle / 7 = index knuckle / 11 = middle knuckle
    # 15 = ring knuckle / 19 = little knuckle
    # while loop until hand is on medication box
    global medication_taken
    while medication_taken == False:
        success, img = webcamCapture2.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    # localise coordinates in the image instead of ratios
                    localiseX, localiseY = int(lm.x * w), int(lm.y * h)
                    # print(id, localiseX, localiseY)
                    if id == 3 or 7 or 11 or 15 or 19:
                        # outputs x and y value of each finger key point
                        finger_coordinates = [localiseX, localiseY]
                        # adds the array to the list
                        # print(localiseX)
                        print(localiseY)
                        fingerKnucklePosition.append(finger_coordinates)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                if minX <= localiseX <= maxX:
                    print ("***********1***********")
                    # print('hand X on box')
                    if minY <= localiseY <= maxY:
                        # hand on box
                        print("***********2***********")
                        print('hand Y on box')
                        medication_taken = True

                if minY <= localiseY <= maxY:
                    print("***********3***********")
                    if minX <= localiseX <= maxX:
                        # hand on box
                        print("***********4***********")
                        print('hand Y on box')
                        medication_taken = True
                else:
                    # recursive hand detection loop
                    handDetection(box_coordinates, minX, minY, maxX, maxY)

        cv2.imshow("Hands", img)


def failLoop(detection_tolerance):
    print("FAIL")
    if detection_tolerance == 0:
        sendAlert(2)
        # recursive loop until medication is taken
        medDetection(5)
    else:
        detection_tolerance = detection_tolerance - 1
        medDetection(detection_tolerance)


def meanOfArray(array):
    return math.trunc(sum(array) / len(array))

def medDetection(detection_tolerance):
    print("medDetection")  ##### remove this when completed
    objectDetector(repeat_tolerance, configPath, weightPath, "basic", "coco.names")
    print(class_identifier)  ##### remove this when completed
    # medication bottle class ID
    # checks if any part of the list has the relevant class
    if 1 in class_identifier:
        print("+----------------------------------------------+")
        print('|               Person detected.               |')
        # print(person_coordinates)
        print("|______________________________________________|")
        #  objectDetector(repeat_tolerance, configPath_medBottles, weightsPath_medBottles, "medication",
        #                 "medbottle.names")
        if 44 in class_identifier:
            print("+----------------------------------------------+")
            print("|             Medication detected.             |")
            # print(box_coordinates)
            print("|______________________________________________|")
            # using average of the data will remove outliers as well as make calculations easier
            cv2.destroyWindow('TEST')
            minX = meanOfArray(minX_box_coordinates)
            print('min X min Y')
            print(minX)
            minY = meanOfArray(minY_box_coordinates)
            print (minY)
            maxX = minX + meanOfArray(width_box_coordinates)
            print(maxX)
            maxY = minY + meanOfArray(height_box_coordinates)
            print (maxY)
            print ("XXXX")
            handDetection(box_coordinates, minX, minY, maxX, maxY)
            # will only pass this function if broken within
            global medication_taken
            medication_taken = True
        else:
            failLoop(detection_tolerance)
    else:
        failLoop(detection_tolerance)


def resetArrays():
    global class_identifier
    class_identifier = []
    global person_coordinates
    person_coordinates = []
    global box_coordinates
    box_coordinates = []
    global minX_box_coordinates
    minX_box_coordinates = []
    global minY_box_coordinates
    minY_box_coordinates = []
    global maxX_box_coordinates
    maxX_box_coordinates = []
    global maxY_box_coordinates
    maxY_box_coordinates = []
    global fingerKnucklePosition
    fingerKnucklePosition = []


def _main_(medication_time):
    resetArrays()
    if medication_time in schedule:
        global medication_taken
        medication_taken = False
        while not medication_taken:
            medDetection(detection_tolerance)
    else:
        # recursive loop until turned off
        _main_()


current_time = now.strftime("%H:%M")
# print("Current Time =", current_time)
# scheduling for medication times.
# print(current_time)
schedule = ["11:11"]


def test():
    while not medication_taken:
        medDetection(detection_tolerance)
    result.release()


def gui():
    print("+----------------------------------------------+")
    print("| Current Time =", current_time, "                        |")
    print("| Please enter a number to select the option:  |")
    print("| 1. Run Program                               |")
    print("| 2. Show Schedule                             |")
    print("| 3. Add to Schedule                           |")
    print("| 4. Remove from Schedule                      |")
    print("| 5. Close                                     |")
    print("|______________________________________________|")
    gui_decision = ""
    while gui_decision != "5":
        gui_decision = input("| Choice: ")
        if gui_decision == "1":  # run program
            test()
            #  _main_("11:11")
        if gui_decision == "2":  # show schedule
            print("+----------------------------------------------+")
            print(schedule)
            print("|______________________________________________|")
        if gui_decision == "3":  # Add to Schedule
            pass
        if gui_decision == "4":  # Remove from Schedule
            pass
        if gui_decision == "5":  # close
            break


# handDetection()
test()
gui()

#############################################################
#  copying
#  print("+----------------------------------------------+")
#  print("|______________________________________________|")

#############################################################
