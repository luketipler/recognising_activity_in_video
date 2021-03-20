# imports initialization
import cv2
from playsound import playsound
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M")
print("Current Time =", current_time)

# from object_detector import *
# from skeletal_tracker import skeletalEstimator
medication_taken = False
# webcam initialization
webcamCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcamCapture.set(3, 480)
webcamCapture.set(4, 360)
# sample files for testing
# webcamCapture = cv2.VideoCapture("pillTaking1.mp4")
# coco class names initialization
classNames = []
# confidence threshold value
threshold = 0.6
class_identifier = []
person_coordinates = []
box_coordinates = []
# path locations for object detection
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
# trained model for medication bottles
configPath_medBottles = 'bottles_label_map.pbtxt'
weightsPath_medBottles = '.pb'
# neural network initializations
# model initialization of neural network for object detection
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(480, 360)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# how many times to run through detection loop, change for more loops
repeat_tolerance = 10
detection_tolerance = 5
# probably not use this
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
# output to mp4 file for testing and demonstration purposes
output_filename = 'test_output.mp4'
output_frames_per_second = 20.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file_size = (1080, 720)
result = cv2.VideoWriter(output_filename,
                         fourcc,
                         output_frames_per_second,
                         file_size)


# alert sending function
def sendAlert(alert_number):
    if alert_number == 1:
        # test sound
        playsound('test.mp3')
    if alert_number == 2:
        # "time to take your medication"
        playsound('')
    if alert_number == 3:
        # "have you taken your medication?"
        playsound('')
    # if alert_number == x: , allow for additional sounds.


def objectDetector(n, weights_path_od, config_path_od, model, class_file):
    global classFile
    classFile = class_file
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    print(classNames)
    # while True:
    for i in range(n):
        success, img = webcamCapture.read()
        global net
        net = cv2.dnn_DetectionModel(weights_path_od, config_path_od)
        net.setInputSize(480, 360)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        class_ids, confidence_values, bounding_box = net.detect(img, threshold)
        print("Class ID = " + str(class_ids)  # + " || ", "Confidence =" + str(confidence_values)
              + "||", "Coordinates " + str(bounding_box))

        if len(class_ids) != 0:
            for classId, confidence, box in zip(class_ids.flatten(), confidence_values.flatten(), bounding_box):
                cv2.rectangle(img, box, color=(106, 13, 173), thickness=2, lineType=None, shift=None)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 25),
                            cv2.FONT_ITALIC, 1, (106, 13, 173), 2)
                class_identifier.append(classId)
                print(class_identifier)
                # add the box coordinates to the list.
                if model == "medication":
                    if classId == 1:
                        box_coordinates.append(box)
                        # print(box_coordinates)
                if model == "basic":
                    if classId == 1:
                        person_coordinates.append(box)
                        # print(box_coordinates)
        cv2.imshow('TEST', img)
        result.write(img)
        cv2.waitKey(10)


def poseEstimation(n):
    for i in range(n):
        break


def medDetection(detection_tolerance):
    print("HEAD")
    objectDetector(repeat_tolerance, configPath, weightsPath, "basic", "coco.names")
    print(class_identifier)
    # medication bottle class ID
    # checks if any part of the list has the relevant class
    if 1 in class_identifier:
        print('|***************************|')
        print('|      person detected      |')
        # print(person_coordinates)
        print('|***************************|')
        objectDetector(repeat_tolerance, configPath_medBottles, weightsPath_medBottles, "medication", "medbottle.names")
        if 77 in class_identifier:
            global medication_taken
            medication_taken = True
            print('|***************************|')
            print("| Medication has been taken |")
            # print(box_coordinates)
            print('|***************************|')
        else:
            print("FAIL")
            if detection_tolerance == 0:
                sendAlert(1)
                detection_tolerance = 5
                # recursive loop until medication is taken
                medDetection(detection_tolerance)
            else:
                detection_tolerance = detection_tolerance - 1
                medDetection(detection_tolerance)
    else:
        print("FAIL")
        if detection_tolerance == 0:
            sendAlert(1)
            detection_tolerance = 5
            # recursive loop until medication is taken
            medDetection(detection_tolerance)
        else:
            detection_tolerance = detection_tolerance - 1
            medDetection(detection_tolerance)


def _main_(medication_time):
    if medication_time == True:
        global medication_taken
        medication_taken = False
        while not medication_taken:
            medDetection(detection_tolerance)
    else:
        # recursive loop until turned off
        _main_()


# scheduling for medication times.
print(current_time)
schedule = []


def test():
    while not medication_taken:
        medDetection(detection_tolerance)
    result.release()


test()
