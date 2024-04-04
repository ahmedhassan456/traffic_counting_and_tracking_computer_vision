import cv2
from tracker.sort import *
import cvzone
from ultralytics import YOLO
import numpy as np


# avilable classes in yolov8
avilableClasses = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# load yolo v8 nano model
model = YOLO('yolo_weights\yolov8n.pt')

# store the id of each car
totalCount = []

# read the video
vid = cv2.VideoCapture('video\cars.mp4')

# get video width and height
videoWidth = int(vid.get(3))
videoheight = int(vid.get(4))
print(f'width = {videoWidth}')
print(f'height = {videoheight}')

# load the mask
mask = cv2.imread('masks\mask.png')

# create tracker object
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# create line for checking if the car overshooted the line or not
limits = [405, 297, 673, 297]

while vid.isOpened():
    _, frame = vid.read()

    # apply mask to the frame and find region of interest
    roi = cv2.bitwise_and(frame, mask)

    # detect the objects
    results = model(roi, stream=True)

    # to store the detections
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            classIndex = int(box.cls[0])
            conf = int(box.conf[0])
            currentClass = avilableClasses[classIndex]

            # detect only cars or trucks or buses or motorcycles 
            if currentClass == 'bus' or currentClass == 'car' or currentClass == 'motorcycle' or currentClass == 'truck':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    # update tracker
    trackerResults = tracker.update(detections)

    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), thickness=5)

    for res in trackerResults:
        # find coordinates of the bounding box
        x1, y1, x2, y2, id = res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1

        # draw the bounding box 
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, colorR=(255,0,255))
        cvzone.putTextRect(frame, f'{int(id)}', (max(0,int(x1)), max(35, int(y1))), scale=2, thickness=1,offset=5)

        # find the center of the bounding box and draw a circle there
        cx, cy = x1 + w //2 , y1 + h //2
        cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        # check if the car overshooted the line or not
        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[3]+15:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), thickness=5)

    # show the total count of cars
    cvzone.putTextRect(frame, f'Count = {len(totalCount)}', (0, 40))

    # show the frame
    cv2.imshow('car counter', frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
vid.release()