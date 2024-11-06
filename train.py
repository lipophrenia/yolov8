import argparse
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser(
    description="""This script trains YOLOv8 model.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-m", "--model", required=True, help="Relative path to model file. Specify 'yolov8{n/s/m/l/x}.pt' to load pretrained model from ultralytics."\
                    " It will be automatically downloaded during first use. Specify 'yolov8n.yaml' to build new model.")
parser.add_argument("-d", "--dataset", required=True, help="Relative path to dataset .yaml file.")
parser.add_argument("-n", "--name", required=True, help="Name of folder for saving progress in ./save")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of train epochs.")
parser.add_argument("-t", "--tw_model", default=None, help="Model whose weights will be transferred. Specify if building new model.")
parser.add_argument("-b", "--batch", type=int, default=-1, help="Batch size. Indicating how many images are processed before the model's internal parameters are updated.")
args = parser.parse_args()

MODEL = args.model
TW_MODEL = args.tw_model
DATASET = args.dataset
NAME = args.name
BATCH = args.batch
EPOCHS = args.epochs

if not os.path.exists('./save'):
        os.makedirs('./save')

if MODEL.endswith('.yaml'):
    if TW_MODEL != None :
        print("\033[1m\033[33mBuilding a new model from YAML, transfer pretrained weights to it and start training.\033[0m\033[22m")
        run = YOLO(MODEL).load(TW_MODEL)
    else:
        print("\033[1m\033[33mBuilding a new model from YAML.\033[0m\033[22m")
        run = YOLO(MODEL)
else :
    print("\033[1m\033[33mStart training from a pretrained *.pt model.\033[0m\033[22m")
    run = YOLO(MODEL)

results = run.train(
    task = 'detect',
    data = DATASET,
    project = './save',
    name = NAME,
    verbose = True,
    cache = True, # Caching of dataset images in RAM
    device = 0, # Single GPU (device=0), multiple GPUs (device=0,1), CPU (device=cpu)

    #base params
    batch = BATCH,
    epochs = EPOCHS,
    time = None, # Overrides the epochs argument, allowing training to automatically stop after the specified duration.
    resume = False, # Resumes training from the last saved checkpoint.
    imgsz = [640,640], # Image size for training. All images will be resized to this dimension before being fed into the model.
    rect=True,
    val = True, # Enables validation during training.
    plots = True, # Generates and saves plots of training and validation metrics.

    #fine-tune
    optimizer = 'auto', # Optimizer to use. SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc. or auto for automatic selection based on model conf.
    lr0=0.01, # Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
    lrf=0.01, # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
    momentum=0.937, # Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.
    cos_lr = False, # Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs.
    weight_decay = 0.0005, # L2 regularization term, penalizing large weights to prevent overfitting.
    box=7.5, # Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.
    cls=0.5, # Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
    dfl=1.5, # Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.
    close_mosaic = 10, # Disables mosaic data augmentation in the last N epochs.
    warmup_epochs=3.0, # Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
    warmup_momentum=0.8, # Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.
    warmup_bias_lr=0.1, # Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.
    freeze = None, # Freezes the first N layers of the model or specified layers by index. Useful for fine-tuning or transfer learning.
    overlap_mask = True # Determines whether segmentation masks should overlap during training.
)

# DEFAULT VALUES

# batch = -1,
# epochs = 100,
# time = None,
# resume = False,
# imgsz = 640,
# val = True,
# plots = True,
# optimizer = 'auto',
# lr0=0.01,
# lrf=0.01,
# momentum=0.937,
# cos_lr = False,
# weight_decay = 0.0005,
# box=7.5,
# cls=0.5,
# dfl=1.5,
# close_mosaic = 10,
# warmup_epochs=3.0,
# warmup_momentum=0.8,
# warmup_bias_lr=0.1,
# freeze = None,
# overlap_mask = True