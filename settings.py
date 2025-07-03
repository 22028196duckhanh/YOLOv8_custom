DATASET_PATH = './foggy_anything' 

RUNS_DIR = './runs_server'

DATASET_YAML_NAME = 'foggy.yaml'
CLASSES = ['person', 'car']
NC = len(CLASSES)

BASE_MODEL = 'yolov8x.pt'

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 20
DEVICE = 0 
PROJECT_NAME = 'yolov8x_final'