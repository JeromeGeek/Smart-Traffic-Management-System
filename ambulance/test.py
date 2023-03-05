from keras.preprocessing.image import img_to_array
from keras.models import load_model
import RPi.GPIO as GPIO
import time
import numpy as np
import imutils
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

channel = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(channel, GPIO.IN)


def callback(channel):
    if GPIO.input(channel):
        return 1
    else:
        return 1


GPIO.add_event_detect(channel, GPIO.BOTH, bouncetime=300)
sound1 = GPIO.add_event_callback(channel, callback)


def init():
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    GPIO.setup(22, GPIO.OUT)
    GPIO.setup(23, GPIO.OUT)
    GPIO.setup(24, GPIO.OUT)


# To stop the mortar set the output to 0 or try to give the reverse direction to wheels so that the wheels won't move.


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)                    # programming the GPIO by BCM pin numbers

TRIG = 18
ECHO = 27

m11 = 16
m12 = 12
m21 = 21
m22 = 20

GPIO.setup(TRIG,GPIO.OUT)                  # initialize GPIO Pin as outputs
GPIO.setup(ECHO,GPIO.IN)                   # initialize GPIO Pin as input

GPIO.setup(m11,GPIO.OUT)
GPIO.setup(m12,GPIO.OUT)
GPIO.setup(m21,GPIO.OUT)
GPIO.setup(m22,GPIO.OUT)

#GPIO.output(led, 1)

time.sleep(5)


def stop():
    print("stop")
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)


def forward():
    GPIO.output(m11, 1)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 0)
    print("Forward")


def back():
    GPIO.output(m11, 0)
    GPIO.output(m12, 1)
    GPIO.output(m21, 0)
    GPIO.output(m22, 1)
    print("back")


def left():
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 0)
    print("left")


def right():
    GPIO.output(m11, 1)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)
    print("right")


def image(camera, model):
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    print("[INFO] loading network...")

    # load the image
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        orig = image.copy()
        image = cv2.resize(image, (32, 32))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        (notAmbulance, ambulance) = model.predict(image)[0]

        # build the label
        label = "ambulance" if (ambulance > notAmbulance and ambulance > 0.7) else "Not Ambulance"
        proba = ambulance if (ambulance > notAmbulance and ambulance > 0.7) else notAmbulance

        label = "{}: {:.2f}%".format(label, proba * 100)

        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        #rawCapture.truncate(0)
        # show the output image
        cv2.imshow("Output", output)

        if label == "ambulance":
            return 1
        else:
            return 0


stop()
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
model = load_model("ambulance_final_32_final.model")

count = 0
while True:
    i = 0
    #avgDistance=0
    #for i in range(5):
    GPIO.output(TRIG, False)                 #Set TRIG as LOW
    time.sleep(0.1)                                   #Delay

    GPIO.output(TRIG, True)                  #Set TRIG as HIGH
    time.sleep(0.00001)                           #Delay of 0.00001 seconds
    GPIO.output(TRIG, False)                 #Set TRIG as LOW

    while GPIO.input(ECHO) == 0:              #Check whether the ECHO is LOW
        continue
       #GPIO.output(led, False)
    pulse_start = time.time()

    while GPIO.input(ECHO) == 1:              #Check whether the ECHO is HIGH
        continue

    pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start  #time to get back the pulse to sensor

    distance = pulse_duration * 17150        #Multiply pulse duration by 17150 (34300/2) to get distance
    distance = round(distance, 2)                 #Round to two decimal points

    flag = 0
    if distance < 15:      #Check whether the distance is within 15 cm range
        count = count+1
        stop()
        time.sleep(1)
        back()
        time.sleep(1.5)
        if (count % 3 == 1) & (flag == 0):
            right()
            flag = 1
        else:
            left()
            flag = 0
            time.sleep(1.5)
            stop()
            time.sleep(1)
    else:
        forward()
        flag = 0

    resultImage = image(camera, model)
    if resultImage == 1:
        stop()
        time.sleep(1)
        left()
        time.sleep(2)
        forward()
