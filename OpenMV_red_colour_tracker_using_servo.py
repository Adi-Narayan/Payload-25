import time
from pyb import Servo, LED
import sensor, image

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
sensor.set_vflip(True)

red = LED(1)
green = LED(2)
blue = LED(3)

#red.on()
#green.on()
#blue.on()

tilt_servo = Servo(2)  # p8

TILT_GAIN = 0.025
SCAN_SPEED = 10
CENTER_Y = sensor.height() // 2

tilt_position = 0
frames_without_detection = 0
MAX_MISSING_FRAMES = 5

THRESHOLD = [(30, 100, 15, 127, 15, 127)]

scan_direction = 1  # 1 means moving up, -1 means moving down

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def apply_rule_of_triads(current, error, gain):
    adjustment = error * gain
    return constrain(current + adjustment, -90, 90)

while True:
    img = sensor.snapshot()
    blobs = img.find_blobs(THRESHOLD, pixels_threshold=100, area_threshold=100)

    if blobs:
        largest_blob = max(blobs, key=lambda b: b.pixels())
        img.draw_rectangle(largest_blob.rect())
        img.draw_cross(largest_blob.cx(), largest_blob.cy())

        error_y = largest_blob.cy() - CENTER_Y
        tilt_position = apply_rule_of_triads(tilt_position, error_y, TILT_GAIN)

        frames_without_detection = 0
    else:
        frames_without_detection += 1
        if frames_without_detection >= MAX_MISSING_FRAMES:
            tilt_position += SCAN_SPEED * scan_direction

            if tilt_position >= 90 or tilt_position <= -90:
                scan_direction *= -1

    tilt_position = constrain(tilt_position, -90, 90)
    tilt_servo.angle(tilt_position)

    print("Tilt Position:", tilt_position, "Frames Without Detection:", frames_without_detection)

    time.sleep_ms(10)
