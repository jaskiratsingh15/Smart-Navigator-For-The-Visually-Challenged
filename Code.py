import os
import cv2
import numpy as np
import time
from picamera2 import Picamera2

last_spoken_time = {}

def NMS(boxes, class_ids, confidences, overlapThresh = 0.5):

    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

    if len(boxes) == 0:
        return [], [], []

    x1 = boxes[:, 0] - (boxes[:, 2] / 2)
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        temp_indices = indices[indices != i]
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    return boxes[indices], class_ids[indices], confidences[indices]


def get_outputs(net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    outs = [c for out in outs for c in out if c[4] > 0.5]
    return outs


def draw(bbox, img):

    xc, yc, w, h = bbox
    img = cv2.rectangle(img,
                        (xc - int(w / 2), yc - int(h / 2)),
                        (xc + int(w / 2), yc + int(h / 2)),
                        (0, 255, 0), 20)

    return img

def speak_label(label, cooldown=5):
    now = time.time()

    if label not in last_spoken_time or (now - last_spoken_time[label] > cooldown):
        print(f"Speaking: {label} detected ahead")
        os.system(f'espeak "{label} detected ahead!"')
        last_spoken_time[label] = now

def get_direction(slopes):
    if not slopes:
        return "No turn (no lines)"
    
    left = sum(1 for s in slopes if s < -0.5)
    right = sum(1 for s in slopes if s > 0.5)
    
    if left > right:
        speak_label("Left Turn")
        return("Left Turn")
    elif right > left:
        speak_label("Right Turn")
        return("Right Turn")
    else:
        speak_label("No Turn")
        return("No Turn")


f=open("Check_py_runs.txt","w")
f.write("Ok")
f.close()

model_cfg_path = '/home/pi/Desktop/yolov3-tiny.cfg'
model_weights_path = '/home/pi/Desktop/yolov3-tiny.weights'
class_names_path = '/home/pi/Desktop/coco.names'
picam2 = Picamera2()
picam2.start()

with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

while True:
    img = picam2.capture_array()
    
    if img.shape[2] == 4:
    	img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    H, W, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), True)

    net.setInput(blob)

    detections = get_outputs(net)

    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    bboxes, class_ids, scores = NMS(bboxes, class_ids, scores)

    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        cv2.putText(img,
                    class_names[class_ids[bbox_]],
                    (int(xc - (w / 2)), int(yc + (h / 2) - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    4)
        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                         2)
        speak_label(class_names[class_ids[bbox_]])
        
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=50)

    slopes = []
    line_lst = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5 or abs(slope) > 2.5:
                continue
            slopes.append(slope)
            for i in slopes:
            	if np.abs(slopes[-1]-i)<0.5:
                    del slopes[-1]
                else:
            		cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    direction = get_direction(slopes)
    cv2.putText(img, f"Direction: {direction}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
	
        

    cv2.imshow("Live Feed", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(1) & 0xFF==27:
        break

cv2.destroyAllWindows()
