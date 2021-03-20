import cv2
from playsound import playsound

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
# model initialization of neural network for object detection
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(480, 360)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# how many times to run through detection loop, change for more loops
repeat_tolerance = 10
detection_tolerance = 5
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




def objectDetector(n):
    # while True:
    for i in range(n):
        success, img = webcamCapture.read()
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
                if classId == 1:
                    box_coordinates.append(box)
                    # print(box_coordinates)

        cv2.imshow('TEST', img)
        result.write(img)
        cv2.waitKey(10)


def poseEstimation(n):
    for i in range(n):
        break
        
        
def medDetection(detection_tolerance):
    print("HEAD")
    objectDetector(repeat_tolerance)
    print(class_identifier)
    # medication bottle class ID
    # checks if any part of the list has the relevant class
    if 1 in class_identifier:
        print('|***************************|')
        print('|      person detected      |')
        # print(person_coordinates)
        print('|***************************|')
        objectDetector(repeat_tolerance)
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
        # poseEstimation(repeat_tolerance)
        # left arm point and right arm point
        # left_hand = ''
        # right_hand = ''
        # face_point = ''
        # med_box_coord = ''
        # if left_hand or right_hand == med_box_coord: # this is for agile sprint 2
        # if left_hand or right_hand == face_point: # this is for agile sprint 3
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


def test():
    while not medication_taken:
        medDetection(detection_tolerance)
    result.release()


test()
