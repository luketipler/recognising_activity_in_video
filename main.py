# imports initialization
import cv2
import mediapipe as mp
from playsound import playsound
from datetime import datetime
import math
import tkinter as tk
import time
import numpy as np
from twilio.rest import Client


##########################################################################
##########################################################################
##########################################################################
##########################################################################
#                                                                        #
# Enter values for your schedule here,                                   #
# These values must be in 24hour "HH:MM" format,                         #
# Example "10:40" or "18:50"                                             #
#                                                                        #
schedule = ["09:00", "11:00", "12:30", "13:50", "19:40", "22:58", "23:03"]
#                                                                        #
##########################################################################
#                                                                        #
# Please insert mp3 files of notifications into files,                   #
# Then add their names into the template below.                          #


def sendAlert(alert_number):                                             #
    global alertRepeat                                                   #
    if alertRepeat == 10:                                                #
        resetArrays()                                                    #
        main()                                                           #
    if alert_number == 1:                                                #
        # test sound                                                     #
        playsound('sound files/test.mp3')                                #
        alertRepeat = alertRepeat + 1                                    #
    if alert_number == 2:                                                #
        # "time to take your medication"                                 #
        playsound('sound files/time to take your medication.mp3')        #
        client.api.account.messages.create(to="+xxxxxxxxxxx", from_="+16027865091",
                                           body="Time to take your medication!")
        alertRepeat = alertRepeat + 1                                    #
    if alert_number == 3:                                                #
        # "have you taken your medication?"                              #
        playsound('sound files/have you taken your medication.mp3')      #
        client.api.account.messages.create(to="+xxxxxxxxxxx", from_="+16027865091",
                                           body="Have you taken your medication?")
        alertRepeat = alertRepeat + 1                                    #
#                                                                        #
#                               Template                                 #
#   if alert_number == x:                                                #
#       playsound('sound files/mp3 Name.mp3')                            #
#       alertRepeat = alertRepeat + 1                                    #
#                                                                        #
##########################################################################
#                                                                        #
#   Please leave the rest of the code unedited to perform effectively.   #
#                                                                        #
##########################################################################
##########################################################################
##########################################################################
##########################################################################


# get current time
now = datetime.now()
# webcam initialization
webcamCapture = cv2.VideoCapture(0)
webcamCapture.set(3, 480)
webcamCapture.set(4, 360)
# coco class names initialization
classNames = []
class_identifier = []
# med bottles names initialization
class_ids = []
confidences = []
boxes = []
# init for params
medication_taken = False
bottle_interaction = False
# confidence threshold value
threshold = 0.6
# how many times to run through detection loop, change for more loops
repeat_tolerance = 10
detection_tolerance = 10
handToFace_tolerance = 250
alertRepeat = 0
count = 10
# array of coordinates for calculations
person_coordinates = []
box_coordinates = []
# min and max values for localisation
minX_box_coordinates = []
minY_box_coordinates = []
width_box_coordinates = []
height_box_coordinates = []
fingerKnucklePosition = []
# path locations for object detection
configPath = 'config files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'config files/frozen_inference_graph.pb'
# trained model for medication bottles
configPath_medBottles = 'config files/custom-yolov4-detector.cfg'
weightsPath_medBottles = "config files/custom-yolov4-detector_best.weights"
# hand media pipe init
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, threshold, threshold)
# pose media pipe init
mpPose = mp.solutions.pose
pose = mpPose.Pose(False, True, True, threshold, threshold)
mpDraw = mp.solutions.drawing_utils
# neural network initializations
# model initialization of neural network for object detection
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(480, 360)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

colour = (106, 13, 173)

# alert sending tokens
account_sid = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
auth_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
client = Client(account_sid, auth_token)


def objectDetector(repeat_tolerance, class_file):
    global classFile
    classFile = class_file
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    for i in range(repeat_tolerance):
        # take webcam feed
        success, img = webcamCapture.read()
        class_ids, confidence_values, bounding_box = net.detect(img, threshold)

        if len(class_ids) != 0:
            for classId, confidence, box in zip(class_ids.flatten(), confidence_values.flatten(), bounding_box):
                cv2.rectangle(img, box, color=(106, 13, 173), thickness=2, lineType=None, shift=None)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 25),
                            cv2.FONT_ITALIC, 1, (106, 13, 173), 2)
                class_identifier.append(classId)

        cv2.imshow('Object Detection', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def bottlesDetector(repeat_tolerance, class_file):
    # initialise the YOLOv4 detection
    netYolo = cv2.dnn.readNet(configPath_medBottles, weightsPath_medBottles)
    global classFile
    classFile = class_file
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    # get layer names using inbuilt function
    layer_names = netYolo.getLayerNames()
    # lopp through and obtain output layers
    for i in netYolo.getUnconnectedOutLayers():
        outputlayers = [layer_names[i[0] - 1]]
    for i in range(repeat_tolerance):
        success, img = webcamCapture.read()
        # take the shape of the image and apply to values
        height, width, channels = img.shape
        # create a blob for use in YOLO from image
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        # set the input of the network to the blob
        netYolo.setInput(blob)
        outs = netYolo.forward(outputlayers)
        # runs on singular output from every output
        for out in outs:
            # for every detection in each, run through
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # if system is confident enough, assign values to arrays
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - w / 2)
                    # assign all relevant detections to arrays
                    confidences.append(float(confidence)),boxes.append([x, y, w, h]), class_ids.append(class_id)
            # non maximum suppression given boxes and corresponding scores
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
            for i in range(len(boxes)):
                if i in indexes:
                    # take the relevant values from array.
                    x, y, w, h = boxes[i]
                    label = str(classNames[class_ids[i]])
                    confidence = confidences[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
                    cv2.putText(img, label + " " + str(confidence), (x, y + 25), cv2.FONT_ITALIC, 1, colour, 2)
                    # assign x,y,w,h values only for bottles.
                    if label == 'bottles':
                        minX_box_coordinates.append(x), minY_box_coordinates.append(y)
                        width_box_coordinates.append(w), height_box_coordinates.append(h)

        cv2.imshow('Medication Detector', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def handDetection(box_coordinates, minX, minY, maxX, maxY, handDetectionLoop):
    # 3 = thumb knuckle / 7 = index knuckle / 11 = middle knuckle
    # 15 = ring knuckle / 19 = little knuckle
    # while loop until hand is on medication box
    global bottle_interaction
    while bottle_interaction == False:
        # take webcam feed
        success, handimg = webcamCapture.read()
        # change the colour format to RGB from BGR
        img_rgb = cv2.cvtColor(handimg, cv2.COLOR_BGR2RGB)
        # store these as results.
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # hand landmarks if results come through as detected
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = handimg.shape
                    # localise coordinates in the image instead of ratios
                    # otherwise the format is ratio of pixels
                    localiseX = int(lm.x * w)
                    localiseY = int(lm.y * h)
                    # chooses relevant points, based on hands
                    if id == 3 or 7 or 11 or 15 or 19:
                        # outputs x and y value of each finger key point
                        finger_coordinates = [localiseX, localiseY]
                        # adds the array to the list
                        fingerKnucklePosition.append(finger_coordinates)
                    # draws the lands marks, and then uses MP packages to connect the dots
                    mpDraw.draw_landmarks(handimg, handLms, mpHands.HAND_CONNECTIONS)
                # uses the box coords to match if hands are inside bounding boxes of medication
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
                    if handDetectionLoop == 0:
                        # send a have you taken your medication alert
                        sendAlert(3)
                        handDetectionLoop = 250
                    else:
                        handDetectionLoop = handDetectionLoop - 1
                    handDetection(box_coordinates, minX, minY, maxX, maxY, handDetectionLoop)

        if handDetectionLoop == 0:
            # send a have you taken your medication alert
            sendAlert(3)
            handDetectionLoop = 250
        else:
            handDetectionLoop = handDetectionLoop -1

        cv2.imshow("Hand Tracking", handimg)
        cv2.waitKey(1)


def handAndFaceTracking(handDetectionLoop):
    global medication_taken
    while medication_taken == False:
        # take webcam feed
        tracksuc, trackimg = webcamCapture.read()
        img_rgb = cv2.cvtColor(trackimg, cv2.COLOR_BGR2RGB)
        handResults = hands.process(img_rgb)
        poseResults = pose.process(img_rgb)
        # run code if a pose is detected
        if poseResults.pose_landmarks:
            mpDraw.draw_landmarks(trackimg, poseResults.pose_landmarks)
            # if id == 10 or 9 (mouth points)
            for id, lm in enumerate(poseResults.pose_landmarks.landmark):
                h, w, c = trackimg.shape
                # add points of the mouth to use in calculations
                if id == 10:
                    # int for co-ords, x value then times by width to get real point
                    mouthLeftX = (int(lm.x * w))
                    mouthLeftY = (int(lm.y * h))
                if id == 9:
                    mouthRightX = (int(lm.x * w))
                    mouthRightY = (int(lm.y * h))
                if id == 8:
                    headLeftX  = (int(lm.x * w))
                # additional variables for non facing calculations
                if id == 7:
                    headRightX = (int(lm.x * w))
                if id == 5:
                    headLeftTopY = (int(lm.y * h))
                if id == 2:
                    headRigthTopY = (int(lm.y * h))
                if id == 16:
                    wristRightX = (int(lm.x * w))
                    wristRightY = (int(lm.y * h))
                if id == 15:
                    wristLeftX = (int(lm.x * w))
                    wristLeftY = (int(lm.y * h))
                # left 8, right 7, 5,2 top , 10,9, bottom
            # run code if a hand is detected
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
                            handOnMouth(mouthLeftX, localiseX, mouthRightX, mouthLeftY, localiseY, mouthRightY, headRightX,
                                  headLeftX, headRigthTopY, headLeftTopY, wristRightX, wristRightY, wristLeftX, wristLeftY)
                        mpDraw.draw_landmarks(trackimg, handLms, mpHands.HAND_CONNECTIONS)

        if handDetectionLoop == 0:
            # send a have you taken your medication alert
            sendAlert(3)
            handDetectionLoop = 250
        else:
            handDetectionLoop = handDetectionLoop -1
        cv2.imshow("Pose Estimation", trackimg)
        cv2.waitKey(1)


# function for getting the hand in mouth acceptance
def handOnMouth(mouthLeftX, localiseX, mouthRightX, mouthLeftY, localiseY, mouthRightY, headRightX,
                headLeftX, headRigthTopY, headLeftTopY, wristRightX, wristRightY, wristLeftX, wristLeftY):
    global medication_taken
    if mouthLeftX <= localiseX <= mouthRightX:
        if mouthLeftY <= localiseY <= mouthRightY:
            # hand on box
            medication_taken = True

    if mouthLeftY <= localiseY <= mouthRightY:
        if mouthLeftX <= localiseX <= mouthRightX:
            # hand on box
            medication_taken = True

    if headLeftX <= wristRightX <= headRightX:
        if min(headRigthTopY, headLeftTopY) <= wristRightY <= min(mouthLeftY, mouthRightY):
            # hand on box
            medication_taken = True

    # wrist within head for sprint 5

    if min(headRigthTopY, headLeftTopY) <= wristRightY <= min(mouthLeftY, mouthRightY):
        if headLeftX <= wristLeftX <= headRightX:
            # hand on box
            medication_taken = True

    if headLeftX <= wristRightX <= headRightX:
        if min(headRigthTopY, headLeftTopY) <= wristLeftY <= min(mouthLeftY, mouthRightY):
            # hand on box
            medication_taken = True

    if min(headRigthTopY, headLeftTopY) <= wristLeftY <= min(mouthLeftY, mouthRightY):
        if headLeftX <= wristLeftX <= headRightX:
            # hand on box
            medication_taken = True


# fail loop to remove redundant code
def failLoop():
    global count, alertRepeat
    if count == 0:
        sendAlert(2)
        # recursive loop until medication is taken
        count = 10
        alertRepeat = alertRepeat + 1
        if alertRepeat == 10:
            resetArrays()
            main()
        medDetection(10)
    else:
        count = count - 1
        medDetection(10)


def meanOfArray(array):
    return math.trunc(sum(array) / len(array))


def medDetection(detection_tolerance):
    global alertRepeat
    alertRepeat = 0
    objectDetector(detection_tolerance, "config files/coco.names")
    # medication bottle class ID
    # checks if any part of the list has the relevant class
    if 1 in class_identifier:
        # person detected
        print("+----------------------------------------------+")
        print('|               Person detected.               |')
        print("|______________________________________________|")
        bottlesDetector(detection_tolerance, 'config files/obj.names')
        if 2 in class_ids:
            # medication detected
            print("+----------------------------------------------+")
            print("|             Medication detected.             |")
            print("|______________________________________________|")
            # using average of the data will remove outliers as well as make calculations easier
            cv2.destroyWindow('Object Detection')
            # retrieve bounding box real co-ords
            minX = meanOfArray(minX_box_coordinates)
            minY = meanOfArray(minY_box_coordinates)
            maxX = minX + meanOfArray(width_box_coordinates)
            maxY = minY + meanOfArray(height_box_coordinates)
            alertRepeat = 0
            handDetection(box_coordinates, minX, minY, maxX, maxY, handToFace_tolerance)
            # will only pass this function if broken within
            global bottle_interaction
            bottle_interaction = True
            # Hand on Medication.
            print("+----------------------------------------------+")
            print("|             Hand on Medication.              |")
            print("|______________________________________________|")
            cv2.destroyWindow('Hand Tracking')
            alertRepeat = 0
            handAndFaceTracking(handToFace_tolerance)
            # Medication Taken.
            print("+----------------------------------------------+")
            print("|              Medication Taken.               |")
            print("|______________________________________________|")
            # destroy all webcam windows
            cv2.destroyAllWindows()
        else:
            failLoop()
    else:
        failLoop()


# reset all arrays once schedule is active
def resetArrays():
    global class_ids
    class_ids = []
    global confidences
    confidences = []
    global boxes
    boxes = []
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
    global bottle_interaction
    bottle_interaction = False
    global alertRepeat
    alertRepeat = 0
    global count
    count = 10


def main():
    # break any previously open windows
    cv2.destroyAllWindows()
    # recalling of time allows for up to date time to be maintained
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    # allows for the system to show it is working, and not presume error
    print(current_time, " Waiting...")
    if current_time in schedule:
        # send initial alert
        playsound('sound files/time to take your medication.mp3')
        resetArrays()
        global medication_taken
        medication_taken = False
        while not medication_taken:
            medDetection(detection_tolerance)
        # recursive loop to constantly loop through
        # has to wait to ensure it does not repeat back through the loop
        time.sleep(55)
        print(current_time, " Waiting...")
        main()
    else:
        # recursive loop until turned off
        # wait time to avoid recursion limit and unnecessary computing resources
        time.sleep(55)
        main()


current_time = now.strftime("%H:%M")
# print("Current Time =", current_time)
# scheduling for medication times.
# print(current_time)


def test():
    # reset so that no tests remain accurate
    resetArrays()
    while not medication_taken:
        medDetection(detection_tolerance)
    # reset again to repeat tests or start system
    resetArrays()


# text based GUI for testing compared to running // this is not used in final but was used in early development
def gui():
    print("+----------------------------------------------+")
    print("| Current Time =", current_time, "                        |")
    print("| Please enter a number to select the option:  |")
    print("| 1. Run Program                               |")
    print("| 2. Show Schedule                             |")
    print("| 3. Close                                     |")
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
        if gui_decision == "3":  # close
            break


window = tk.Tk()
window.title("Medication Tracker")


def runClick():
    runPage = tk.Tk()
    runPage.title("Run Page")
    main_button = tk.Button(runPage, text="Run System", command=main , bg="pale green")
    main_button.grid(column=0, padx=10, row=0, pady=15)
    test_button = tk.Button(runPage, text="Run Tests", command=test, bg="salmon")
    test_button.grid(column=1, padx=10, row=0, pady=15)


def scheduleClick():
    global schedule
    schedulePage = tk.Tk()
    schedulePage.title("Schedule")
    schedule_button = tk.Label(schedulePage, text=schedule, padx=25, pady=10)
    schedule_button.grid(row=0, column=1)


run_label = tk.Button(window, text="Run Program", command=runClick, bg="pale green")
run_label.grid(row=0, padx=10, column=0, pady=10)
show_schedule_label = tk.Button(text="Show Schedule" , command=scheduleClick, bg='sky blue')
show_schedule_label.grid(row=0, padx=10, column=1, pady=10)
quit_button = tk.Button(window, text="Exit Program", command=window.quit, bg="salmon")
quit_button.grid(row=0, padx=10, column=2, pady=10)
window.mainloop()
