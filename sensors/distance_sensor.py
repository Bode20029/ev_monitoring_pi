import time
import RPi.GPIO as GPIO
from config import DISTANCE_MEASUREMENT_INTERVAL

class DistanceSensor:
    def __init__(self, trig_pin, echo_pin):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin

    def get_distance(self):
        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)

        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        pulse_start = pulse_end = time.time()
        timeout = pulse_start + 1.0

        while GPIO.input(self.echo_pin) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None

        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)

    def run(self, queue, stop_event):
        while not stop_event.is_set():
            dist = self.get_distance()
            if dist is not None:
                if queue.full():
                    queue.get_nowait()  # Remove oldest item if queue is full
                queue.put_nowait(dist)
            time.sleep(DISTANCE_MEASUREMENT_INTERVAL)