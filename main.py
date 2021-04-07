# imports initialization
import cv2
import mediapipe as mp
from playsound import playsound
from datetime import datetime
import math


# get current time
now = datetime.now()
# from object_detector import *
# from skeletal_tracker import skeletalEstimator
medication_taken = False
bottle_interaction = False
# webcam initialization
webcamCapture = cv2.VideoCapture(0)
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
# pose media pipe init
mpPose =mp.solutions.pose
pose = mpPose.Pose(False, True, True, threshold, threshold)
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

        if len(class_ids) != 0:
            for classId, confidence, box in zip(class_ids.flatten(), confidence_values.flatten(), bounding_box):
                cv2.rectangle(img, box, color=(106, 13, 173), thickness=2, lineType=None, shift=None)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 25),
                            cv2.FONT_ITALIC, 1, (106, 13, 173), 2)
                class_identifier.append(classId)
                # add the box coordinates to the list.
                if model == "basic":
                    if classId == 44:
                        box_coordinates.append(box)
                        # [ x, y , w , h ]
                        minX_box_coordinates.append(box[0])
                        minY_box_coordinates.append(box[1])
                        width_box_coordinates.append(box[2])
                        height_box_coordinates.append(box[3])
                # if model == "basic":
                # if classId == 1:
                # person_coordinates.append(box)
                # print(person_coordinates) ##### remove this when completed
        cv2.imshow('Object Detection', img)
        result.write(img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


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
    global bottle_interaction
    while bottle_interaction == False:
        success, img = webcamCapture.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    # localise coordinates in the image instead of ratios
                    localiseX = int(lm.x * w)
                    localiseY = int(lm.y * h)
                    # print(id, localiseX, localiseY)
                    if id == 3 or 7 or 11 or 15 or 19:
                        # outputs x and y value of each finger key point
                        finger_coordinates = [localiseX, localiseY]
                        # adds the array to the list
                        fingerKnucklePosition.append(finger_coordinates)

                if minX <= localiseX <= maxX:
                    if minY <= localiseY <= maxY:
                        # hand on box
                        bottle_interaction = True

                if minY <= localiseY <= maxY:
                    if minX <= localiseX <= maxX:
                        # hand on box
                        bottle_interaction = True
                else:
                    # recursive hand detection loop
                    handDetection(box_coordinates, minX, minY, maxX, maxY)

        cv2.imshow("Hand Tracking", img)
    cv2.destroyAllWindows()


def handAndFaceTracking():
    global medication_taken
    while medication_taken == False:
        tracksuc, trackimg = webcamCapture.read()
        img_rgb = cv2.cvtColor(trackimg, cv2.COLOR_BGR2RGB)
        handResults = hands.process(img_rgb)
        poseResults = pose.process(img_rgb)

        if poseResults.pose_landmarks:
            mpDraw.draw_landmarks(trackimg, poseResults.pose_landmarks)
            # if id == 10 or 9 (mouth points)
            for id, lm in enumerate(poseResults.pose_landmarks.landmark):
                h, w, c = trackimg.shape
                if id == 10:
                    mouthLeftX = (int(lm.x * w))
                    mouthLeftY = (int(lm.y * h))
                if id == 0:
                    mouthRightX = (int(lm.x * w))
                    mouthRightY = (int(lm.y * h))

            if handResults.multi_hand_landmarks:
                for handLms in handResults.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = trackimg.shape
                        # localise coordinates in the image instead of ratios
                        localiseX = int(lm.x * w)
                        localiseY = int(lm.y * h)
                        # print(id, localiseX, localiseY)
                        if id == 4 or 8 or 12 or 16 or 20:
                            # outputs x and y value of each finger key point
                            handOnMouth(mouthLeftX, localiseX, mouthRightX, mouthLeftY, localiseY, mouthRightY)
                        mpDraw.draw_landmarks(trackimg, handLms, mpHands.HAND_CONNECTIONS)


        cv2.imshow("Pose Estimation", trackimg)
        cv2.waitKey(1)


def handOnMouth(mouthLeftX, localiseX, mouthRightX, mouthLeftY, localiseY, mouthRightY):
    if mouthLeftX <= localiseX <= mouthRightX:
        if mouthLeftY >= localiseY >= mouthRightY:
            # hand on box
            global medication_taken
            medication_taken = True

    if mouthLeftY >= localiseY >= mouthRightY:
        if mouthLeftX <= localiseX <= mouthRightX:
            # hand on box
            medication_taken = True



def failLoop(detection_tolerance):
    print("+----------------------------------------------+")
    print('|           Medication not Taken.              |')
    print("|______________________________________________|")
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
    objectDetector(repeat_tolerance, configPath, weightPath, "basic", "coco.names")
    # medication bottle class ID
    # checks if any part of the list has the relevant class
    if 1 in class_identifier:
        print("+----------------------------------------------+")
        print('|               Person detected.               |')
        print("|______________________________________________|")
        #  objectDetector(repeat_tolerance, configPath_medBottles, weightsPath_medBottles, "medication",
        #                 "medbottle.names")
        if 44 in class_identifier:
            print("+----------------------------------------------+")
            print("|             Medication detected.             |")
            print("|______________________________________________|")
            # using average of the data will remove outliers as well as make calculations easier
            cv2.destroyWindow('Object Detection')
            minX = meanOfArray(minX_box_coordinates)
            minY = meanOfArray(minY_box_coordinates)
            maxX = minX + meanOfArray(width_box_coordinates)
            maxY = minY + meanOfArray(height_box_coordinates)
            handDetection(box_coordinates, minX, minY, maxX, maxY)
            # will only pass this function if broken within
            global bottle_interaction
            bottle_interaction = True
            print("+----------------------------------------------+")
            print("|             Hand on Medication.              |")
            print("|______________________________________________|")
            cv2.destroyWindow('Hand Tracking')
            handAndFaceTracking()
            print("+----------------------------------------------+")
            print("|              Medication Taken.               |")
            print("|______________________________________________|")
            cv2.destroyAllWindows()
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
    global width_box_coordinates
    width_box_coordinates = []
    global height_box_coordinates
    height_box_coordinates = []
    global fingerKnucklePosition
    fingerKnucklePosition = []
    global medication_taken
    medication_taken = False


def main(medication_time):
    resetArrays()
    if medication_time in schedule:
        global medication_taken
        medication_taken = False
        while not medication_taken:
            medDetection(detection_tolerance)
    else:
        # recursive loop until turned off
        main()


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
            #  main("10:10")
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
#handAndFaceTracking()
test()
gui()

#############################################################
#  copying
#  print("+----------------------------------------------+")
#  print("|______________________________________________|")

#############################################################
