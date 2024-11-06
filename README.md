# MODEL TRAINING AND CONVERTING FOR ROCKCHIP NPU

```bash
git clone https://github.com/airockchip/ultralytics_yolov8
cd ultralytics_yolov8
python3 -m venv env
source ./env/bin/activate
pip install -e .
pip install lapx pafy # for inference
pip install PyGObject # for gst-rtsp-server with YOLO inference
pip install onnx onnxruntime onnxsim onnxruntime-gpu # for export
mkdir custom_scripts && cd custom_scripts
```
Copy scripts from the root of this repository to `custom_scripts`.

### Learning

```bash
python3 train.py -m <model> -d <dataset> -n <name> -e <epochs> [-t <tw_model>]
# model - relative path to the .pt model, if we want to build a new one, we specify, for example, 'yolov8n.yaml' (you can specify any basic YOLOv8 model).
# dataset - relative path to the dataset .yaml file, e.g. './datasets/VisDrone2019-DET/VisDrone2019-DET.yaml'.
# name - name of the directory for saving learning results, created in './save'.
# epochs - number of training epochs. Default is 100.
# tw_model - relative path to the .pt model whose weights we want to transfer. Specify only if we want to transfer weights to a newly built model.
```

### Convertation for Rockchip

```bash
yolo export model=./save/{name}/weights/last.pt format=rknn simplify
```
In the directory `./save/{name}/weights/` there should be an `.onnx` model. Rename it and put it in `onnx2rknn`.
Place the entire `onnx2rknn` directory next to the cloned repository (https://github.com/airockchip/ultralytics_yolov8) and go into it.
Dependencies for conversion to the required version of `python` are pulled from [here](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages).

```bash
python3 -m venv env
source ./env/bin/activate
pip install -r requirements_cp310-2.2.0.txt
pip install rknn_toolkit2-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python3 convert.py onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)] # possible args values are indicated in the convert.py script
# The model is ready for inference on Rockchip NPU
```

### PC inference

```bash
# base inference
python3 track.py -m <model> -s <source> [-width <width>] [-height <height>] [-iou <iou>] [-conf <conf>]
# model - relative path to .pt model.
# source - source (image/video/rtsp)
# width - source width resolution.
# height - source height resolution.
# iou - intersection over union, float, 0 < iou < 1.
# conf - minimum confidence threshold, float, 0 < conf < 1.
# -h for help, examples of launch commands are at the end of the track.py file
```

```bash
# inference with custom post-process
python3 track_cam.py -m <model> -s <source> [-width <width>] [-height <height>] [-iou <iou>] [-conf <conf>] [-t] [-v] #requires opencv
# model - relative path to .pt model.
# source - source (image/video/rtsp)
# width - source width resolution.
# height - source height resolution.
# iou - intersection over union, float, 0 < iou < 1.
# conf - minimum confidence threshold, float, 0 < conf < 1.
# -t - draw traces of detections
# -v - tensors log
# -h for help, examples of launch commands are at the end of the track.py file
```

```bash
# inference + rtsp stream
python3 rtsp_yolo_server.py -src <source> [-fps <fps>] [-width <width>] [-height <height>] [-port <port>] [-uri <uri>] [-m <model>] [-iou <iou>] [-conf <conf>]
# src,source - video source supported by OpenCV.
# fps - frame rate.
# width - source width resolution.
# height - source height resolution.
# port - port for rtsp server.
# uri - stream uri, specify starting with /.
# model - relative path to .pt model.
# iou - intersection over union, float, 0 < iou < 1.
# conf - minimum confidence threshold, float, 0 < conf < 1.
# -h for help
```
