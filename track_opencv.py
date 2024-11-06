import argparse
import importlib.util
import sys
import numpy as np

# dynamic import opencv-python with GStreamer api support
opencv_gst_path = "/usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("cv2", opencv_gst_path)
cv2 = importlib.util.module_from_spec(spec)
sys.modules["cv2"] = cv2
spec.loader.exec_module(cv2)
print("OpenCV version:", cv2.__version__)

# default import opencv-python without GStreamer api support
# import cv2

from collections import defaultdict
from ultralytics import YOLO

def gen_color(class_num):
    color_list = []
    np.random.seed(1)
    while 1:
        a = list(map(int, np.random.choice(range(255),3)))
        if(np.sum(a)==0): continue
        color_list.append(a)
        if len(color_list)==class_num: break
    return color_list

def set_resolution(width,height):
    return (height,width)

parser = argparse.ArgumentParser(
    description="""YOLOv8 inference.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-m", "--model", required=True, help="relative path to YOLOv8 .pt model")
parser.add_argument("-s","--source", default='/dev/video0', help="source, accepted formats are similar to openCV")
parser.add_argument("-width", "--frame_width", default=1920, help="video frame width", type = int)
parser.add_argument("-height", "--frame_height", default=1080, help="video frame height", type = int)
parser.add_argument("-iou", "--iou_thres", default= 0.5, help="intersection over union threshold", type=float)
parser.add_argument("-conf", "--conf_thres", default=0.25, help="minimum confidence threshold for detections", type=float)
parser.add_argument("-t", "--trace", help="trace detections", action='store_true')
parser.add_argument("-v", "--verbose", help="log tensors", action='store_true')
args = parser.parse_args()

yolo = YOLO(args.model)
CLASSES = ("person", "bicycle", "car","motorbike","aeroplane","bus ","train","truck ","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
            "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife",
            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
            "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
            "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier", "toothbrush")
colorlist = gen_color(len(CLASSES))
fontScale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 1
rect_thickness = 2

cap = cv2.VideoCapture(args.source)
cap.get(cv2.CAP_PROP_BACKEND)
if (cap.get(cv2.CAP_PROP_BACKEND)!=cv2.CAP_GSTREAMER):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    frame_size = set_resolution(args.frame_width, args.frame_height)
else:
    frame_size = set_resolution(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("YOLOv8 Camera",cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Camera", 1600,900)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (args.frame_width, args.frame_height), interpolation = cv2.INTER_LINEAR)

    results = yolo.track(source=frame, imgsz=frame_size, iou=args.iou_thres, conf=args.conf_thres, stream=True, show=False, persist=True, max_det=100)


    for result in results:
        boxes = result.boxes
        if (args.verbose==True):
            print(boxes)
        print("Total detections -",boxes.cls.size(dim=0))
        i=0
        while i<boxes.cls.size(dim=0):
            # print("DETECTION",i)
            # print("class id:",boxes.cls[i].item())
            # print("confidence:", boxes.conf[i].item())
            class_id = round(boxes.cls[i].item())
            x1=round(boxes.xyxy[i][0].item())
            y1=round(boxes.xyxy[i][1].item())
            x2=round(boxes.xyxy[i][2].item())
            y2=round(boxes.xyxy[i][3].item())
            w=round(boxes.xywh[i][2].item())
            h=round(boxes.xywh[i][3].item())

            if boxes.is_track==True and args.trace==True:
                id=round(boxes.id[i].item())
                track = track_history[id]
                track.append((float(x1+w/2), float(y1+h/2)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colorlist[round(boxes.cls[i].item())], thickness=3)
            
            p1=(x1,y1)
            p2=(x2,y2)
            # print("p1 =",p1)
            # print("p2 =",p2)
            cv2.rectangle(frame, p1, p2, colorlist[round(boxes.cls[i].item())], rect_thickness) # main box

            label=f'{CLASSES[class_id]}: {boxes.conf[i].item():.2f}'
            text_size = cv2.getTextSize(label, font, fontScale, font_thickness)[0]
            label_bg_p1 = p1
            label_bg_p2 = (x1+text_size[0],y1+text_size[1]+2);
            cv2.rectangle(frame, label_bg_p1, label_bg_p2, colorlist[round(boxes.cls[i].item())], -1) # label_bg
            
            text_coord=(x1,y1+text_size[1])
            cv2.putText(frame, label, text_coord, font, fontScale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            i+=1

    cv2.imshow("YOLOv8 Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()