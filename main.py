# imports initialization
import sys
import time
import cv2
import numpy
from playsound import playsound

# from object_detector import *
# from skeletal_tracker import skeletalEstimator
medication_taken = False
# webcam initialization
webcamCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcamCapture.set(3, 1920)
webcamCapture.set(4, 1080)
# coco class names initialization
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# confidence threshold value
threshold = 0.6
class_identifier = []
box_coordinates = []
# path locations for object detection
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
# neural network initializations
# model initialization of neural network for object detection
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(1080, 720)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# how many times to run through detection loop, change for more loops
repeat_tolerance = 5


# initialize body parts and pairs of said parts
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


# alert sending function
def sendAlert(alert_number):
    if alert_number == 1:
        # test sound
        playsound('test.mp3')
    if alert_number == 2:
        # "time to take your medication"
        playsound('test.mp3')
    if alert_number == 3:
        # "have you taken your medication?"
        playsound('test.mp3')
    # if alert_number == x: , allow for additional sounds.


def objectDetector(n):
    # while True:
    for i in range(n):
        success, img = webcamCapture.read()
        class_ids, confidence_values, bounding_box = net.detect(img, threshold)
        print("Class ID = " + str(class_ids) # + " || ", "Confidence =" + str(confidence_values)
              + "||", "Coordinates " + str(bounding_box))

        if len(class_ids) != 0:
            for classId, confidence, box in zip(class_ids.flatten(), confidence_values.flatten(), bounding_box):
                cv2.rectangle(img, box, color=(106, 13, 173), thickness=2, lineType=None, shift=None)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 25),
                            cv2.FONT_ITALIC, 1, (106, 13, 173), 2)
                class_identifier.append(classId)
                print(class_identifier)
                if classId == 1:
                    box_coordinates.append(box)
                    print(box_coordinates)
        cv2.imshow('TEST', img)
        cv2.waitKey(5)


def poseEstimation(n):
    for i in range(n):
        break


def medDetection():
    print("HEAD")
    objectDetector(repeat_tolerance)
    print(class_identifier)
    # medication bottle class ID
    # checks if any part of the list has the relevant class
    if any(class_identifier) == 1:
        print('***************')
        print(box_coordinates)
        print('***************')
        poseEstimation(repeat_tolerance)
        # left arm point and right arm point
        # left_hand = ''
        # right_hand = ''
        # face_point = ''
        # med_box_coord = ''
        # if left_hand or right_hand == med_box_coord:
        # if left_hand or right_hand == face_point:
        # medication_taken = True
    else:
        print("FAIL")
        sendAlert(1)
        # if detection_tolerance == 0:
        # sendAlert(1)
        # recursive loop until medication is taken
        # medDetection()
        # else:
        # detection_tolerance = detection_tolerance - 1
        # medDetection()


def _main_(medication_time):
    if medication_time == True:
        medication_taken = False
        while medication_taken == False:
            medDetection()
    else:
        # recursive loop until turned off
        _main_()


medDetection()
