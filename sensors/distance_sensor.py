from config import DISTANCE_MEASUREMENT_INTERVAL
from gpiozero import DistanceSensor
import time


class DistanceSensor:
    def __init__(self, trig_pin, echo_pin):
        self.sensor = DistanceSensor(trigger=trig_pin, echo=echo_pin)

    def get_distance(self):
        try:
            distance = self.sensor.distance * 100  # Convert to cm
            return round(distance, 2)
        except TimeoutError:
            return None

    def run(self, queue, stop_event):
        while not stop_event.is_set():
            dist = self.get_distance()
            if dist is not None:
                if queue.full():
                    queue.get_nowait()  # Remove oldest item if queue is full
                queue.put_nowait(dist)
            time.sleep(DISTANCE_MEASUREMENT_INTERVAL)