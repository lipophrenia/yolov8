import argparse
from ultralytics import YOLO

def srctype(src):
    if (src.startswith('rtsp://') | src.endswith('.asf') | src.endswith('.avi') |
        src.endswith('.gif') | src.endswith('.m4v') | src.endswith('.mkv') |
        src.endswith('.mov') | src.endswith('.mp4') | src.endswith('.mpeg') |
        src.endswith('.mpg') | src.endswith('.ts') | src.endswith('.wmv') | src.endswith('.webm')) :
        return 'stream/video'
    elif (src.endswith('.bmp') | src.endswith('.dng') | src.endswith('.jpeg') |
        src.endswith('.jpg') | src.endswith('.mpo') | src.endswith('.png') |
        src.endswith('.tif') | src.endswith('.tiff') | src.endswith('.webp') | src.endswith('.pfm')) :
        return 'image'
    else :
        return 'unknown'

def set_resolution(width,height):
    return (height,width)

def main(model, src, resolution_w, resolution_h, set_conf, set_iou):
    run = YOLO(model)

    resolution = set_resolution(resolution_w, resolution_h)
    type = srctype(src) 
        
    if type == 'stream/video' :
        results = run.track(src, stream=True, show=True, imgsz=resolution, conf=set_conf, iou=set_iou, max_det=100, show_labels=True, show_conf=True)
        for result in results:
            boxes = result.boxes
    elif type == 'image' :
        results = run.track(src, stream=True, show=False, imgsz=resolution, conf=set_conf, iou=set_iou, max_det=100, show_labels=True, show_conf=True)
        for result in results:
            boxes = result.boxes
            # result.show() # for local show
            result.save(filename='result.jpg')
            print("Total detections -",boxes.cls.size(dim=0))
            i=0
            while i<boxes.cls.size(dim=0):
                print("DETECTION",i)
                print("class id:",boxes.cls[i].item())
                print("confidence:", boxes.conf[i].item())
                print("x1 =",boxes.xyxy[i][0].item())
                print("y1 =",boxes.xyxy[i][1].item())
                print("x2 =",boxes.xyxy[i][2].item())
                print("y2 =",boxes.xyxy[i][3].item())
                i+=1
    elif type == 'unknown' :
        print("Unknown source type. Use one of the following:")
        print("Stream formats: rtsp")
        print("Video formats: asf, avi, gif, m4v, mkv, mov, mp4, mpeg, mpg, ts, wmv, webm")
        print("Image formats: bmp, dng, jpeg, jpg, mpo, png, tif, tiff, webp, pfm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This script runs YOLOv8 .pt model on local PC.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", required=True, help="Relative path to YOLOv8 .pt model.")
    parser.add_argument("-s", "--src", required=True, help="Relative path to video/image file.")
    parser.add_argument("-width", "--resolution_w", type=int, default=640, help="Defines the image width for inference.")
    parser.add_argument("-height", "--resolution_h", type=int, default=640, help="Defines the image height for inference.")
    parser.add_argument("-iou", "--set_iou", type=float, default= 0.5, help="Intersection Over Union threshold. Lower values result in fewer detections by eliminating overlapping boxes.")
    parser.add_argument("-conf", "--set_conf", type=float, default=0.25, help="Sets the minimum confidence threshold for detections.")
    args = parser.parse_args()

    main(args.model, args.src, args.resolution_w, args.resolution_h, args.set_conf, args.set_iou)

# Launch commands examples
# python3 track.py -m ./save/model_name/weights/last.pt -s ./test_data/video.mp4 -conf 0.5
# python3 track.py -m ./save/model_name/weights/last.pt -s ./test_data/img.jpg -iou 0.7
# python3 track.py -m yolov8n.pt -s rtsp://172.16.10.255:554/video.pro1 -iou 0.5 -conf 0.4