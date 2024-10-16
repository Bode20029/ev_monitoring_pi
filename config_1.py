import os
from dotenv import load_dotenv

load_dotenv()

# Constants
DISTANCE_THRESHOLD = 30  # cm
CURRENT_THRESHOLD = 0.1  # A
DETECTION_STABILITY_TIME = 5  # seconds
CHARGING_CHECK_TIME = 600  # 10 minutes
ALARM_INTERVALS = [180, 300, 600]  # 3 minutes, 5 minutes, 10 minutes

# Sensor Constants
TRIG_PIN = 12
ECHO_PIN = 16
DISTANCE_MEASUREMENT_INTERVAL = 0.1
PZEM_PORT = '/dev/ttyUSB0'
PZEM_MEASUREMENT_INTERVAL = 1  # 1 second

# Notifications
LINE_NOTIFY_TOKEN = os.getenv("LINE_NOTIFY_TOKEN")

# Database
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "cardb"
COLLECTION_NAME = "nonev"

# YOLO model
MODEL_PATH = 'models/yolov9s_ev.pt'

# Vehicle Classes
EV_BRANDS = [
    'AION',
    'BYD',
    'DEPAL',
    'HAVAL',
    'MG',
    'NETA',
    'ORA',
    'TESLA',
    'VOLVO',
    'XPENG'
]
VEHICLE_CLASSES = EV_BRANDS + ['NON-EV']
