# required
# sudo apt-get install libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp libgirepository1.0-dev libcairo2-dev

import gi
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

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

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture(args.source)
        self.number_frames = 0
        self.fps = args.fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                            'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                            '! videoconvert ! video/x-raw,format=NV12 ' \
                            '! nvh264enc ! rtph264pay name=pay0 pt=96' \
                            .format(args.frame_width, args.frame_height, self.fps)
    # method to capture the video feed from the camera and push it to the
    # streaming buffer.
    def on_need_data(self, src, length):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # It is better to change the resolution of the camera 
                # instead of changing the image shape as it affects the image quality.
                frame = cv2.resize(frame, (args.frame_width, args.frame_height), interpolation = cv2.INTER_LINEAR)
                
                results = yolo.track(source=frame, imgsz=frame_size, iou=args.iou_thres, conf=args.conf_thres, stream=True, show=False, max_det=100)
                for result in results:
                    boxes = result.boxes
                    # print("Total detections -",boxes.cls.size(dim=0))
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

                data = frame.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames, self.duration, self.duration / Gst.SECOND))
                if retval != Gst.FlowReturn.OK:
                    print(retval)
    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(args.port))
        self.get_mount_points().add_factory(args.stream_uri, self.factory)
        self.attach(None)
        print(f"Stream is running on rtsp://<yourIP>:{args.port}{args.stream_uri}")

parser = argparse.ArgumentParser(
    description="""RTSP server with YOLOv8 inference.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-src","--source", default='/dev/video0', help="source, accepted formats are similar to openCV")
parser.add_argument("-fps","--fps", default=30, help="fps of the camera", type = int)
parser.add_argument("-width", "--frame_width", default=640, help="video frame width", type = int)
parser.add_argument("-height", "--frame_height", default=480, help="video frame height", type = int)
parser.add_argument("-port","--port", default=8554, help="port to stream video", type = int)
parser.add_argument("-uri", "--stream_uri", default = "/camera", help="rtsp video stream uri")
parser.add_argument("-m", "--model", default='yolov8n.pt', help="relative path to YOLOv8 .pt model")
parser.add_argument("-iou", "--iou_thres", default= 0.7, help="intersection over union threshold", type=float)
parser.add_argument("-conf", "--conf_thres", default=0.25, help="minimum confidence threshold for detections", type=float)
args = parser.parse_args()

print("Starting...")

yolo = YOLO(args.model)
frame_size = set_resolution(args.frame_width, args.frame_height)

fontScale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 1
rect_thickness = 1

CLASSES = ("person", "bicycle", "car","motorbike","aeroplane","bus ","train","truck ","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
            "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife",
            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
            "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
            "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier", "toothbrush")
colorlist = gen_color(len(CLASSES))

# initializing the threads and running the stream on loop.
Gst.init(None)
server = GstServer()
loop = GLib.MainLoop()
loop.run()