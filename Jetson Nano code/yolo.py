# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import paho.mqtt.publish as publish
import time
import paho.mqtt.client as paho

MQTT_SERVER = "140.127.196.41"
MQTT_PORT = 18883
MQTT_client = paho.Client("MQTT_client")
MQTT_client.connect(MQTT_SERVER,MQTT_PORT)

import Jetson.GPIO as GPIO
import threading

#LED
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(18,GPIO.OUT)

import serial
import serial.tools.list_ports

import tensorflow as tf
import json

from datetime import datetime
from datetime import date

def flash():
    print("Flash function be loaded into program")
    while True:
        GPIO.output(18,GPIO.HIGH)
        time.sleep(1)
        GPIO.output(18,GPIO.LOW)
        time.sleep(1)

number=0
rftx="rf tx "

class LoRaapi:
    def __init__(self):
        global number
        state=True
        while state==True:
            self.PortList = serial.tools.list_ports.comports()#print port and board
            print("serial port list number == ",len(self.PortList))
            for i in range(len(self.PortList)):
                try:
                    print("list ==", i)
                    print("port num ==", self.PortList[i].device)
                    self.serial_port = serial.Serial(
                        port=self.PortList[i].device,
                        baudrate=115200,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=1.0,
                    )
                except:
                    continue
                else:
                    #set lora
                    '''
                    self.serial_port.write(b'sip factory_reset')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)

                    self.serial_port.write(b'rf set_sync 12')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)

                    self.serial_port.write(b'rf set_freq 922500000')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)

                    self.serial_port.write(b'rf set_sf 10')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)

                    self.serial_port.write(b'rf set_bw 125')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)

                    self.serial_port.write(b'rf save')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)
                    '''
                    
                    self.serial_port.write(b'sip reset')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)
                    
                    self.serial_port.write(b'sip get_hw_model')
                    c = self.serial_port.read_until(b'\n\r>> S76S\n')
                    print(c)
                    '''
                    if self.PortList[i].device == "COM5":
                        self.serial_port.write(b'rf rx_con on')
                        c=self.serial_port.read_until(b'\n\r>> S76S\n')
                        print(c)
                    '''
                    if c == b'\n\r>> S76S\n':
                        number = i
                        state = False
                        break
                finally:
                    self.serial_port.close()
                    print("init() close lora")
    def open(self,lora_num):
        global number,rftx
        self.portnum = serial.tools.list_ports.comports()
        self.port = serial.Serial(
            port=self.portnum[number].device,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5.0,
        )
        data1111=rftx+lora_num + "1111"
        data1111 = str.encode(data1111)
        print(data1111)
        self.port.write(data1111)
        c = self.port.read_until(b'radio_tx_ok')
        print(c)
        self.port.close()
        #print("open() close lora")
    def close(self,lora_num):
        global number, rftx
        self.portnum = serial.tools.list_ports.comports()
        self.port = serial.Serial(
            port=self.portnum[number].device,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5.0,
        )
        data1111 = rftx + lora_num + "1100"
        data1111 = str.encode(data1111)
        print(data1111)
        self.port.write(data1111)
        c = self.port.read_until(b'radio_tx_ok')
        print(c)
        self.port.close()
        #print("close() close lora")

class YOLO(object):
    _defaults = {
        "model_path": '/home/jetson/Desktop/keras-yolo3_nonet/model_data/yolov3-tiny-bird_30000.h5',	#model_data/yolov3-tiny-bird_30000.h5 #V3_h5/yolov3-bird_18000.h5
        "anchors_path": '/home/jetson/Desktop/keras-yolo3_nonet/model_data/tiny_yolo_anchors.txt',	#yolo_anchors.txt
        "classes_path": '/home/jetson/Desktop/keras-yolo3_nonet/model_data/coco_classes_bird.txt',	#coco_classes_bird.txt
        "score" : 0.9,
        "iou" : 0.9,
        "model_image_size" : (416, 416), 
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        today = date.today()
        day = str(today.strftime("%Y/%m/%d"))

        now = datetime.now()
        day_time = now.strftime("%H:%M:%S")

        print()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        if len(out_boxes)>=1:
            appear_bird = True
        else:
            appear_bird = False
        bird_count = len(out_boxes)

        #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        font = ImageFont.truetype(font='/usr/share/fonts/opentype/NotoSerifCJK-Bold.ttc',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        
        end = timer()
        print(end - start)
        return image, appear_bird, bird_count, day, day_time

    def close_session(self):
        self.sess.close()

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=3280,
    display_height=2464,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def send_data(bird_count, day, day_time):
    f=open("/home/jetson/Desktop/keras-yolo3_net/result.jpg", "rb") #3.7kiB in same folder
    fileContent = f.read()
    bird_images = bytearray(fileContent)

    data = {
        'day' : day,
        'time' : day_time,
        'count' : bird_count,
    }
    bird_messenge = json.dumps(data, sort_keys=True, indent=0)

    
    ret_images = MQTT_client.publish("AIoT/bird_images", bird_images)
    ret_messenge = MQTT_client.publish("AIoT/bird_messenge", bird_messenge)
    f.close()

def detect_video(yolo, video_path, output_path=""):

    #vid = cv2.VideoCapture("/home/jetson/Desktop/keras-yolo3_net/farm_videos.MOV")

    #print(gstreamer_pipeline(flip_method=0))
    #vid = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    vid = cv2.VideoCapture(-1)
    
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    #if isOutput:
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), #type(video_size))
        #out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    #threshold variable
    second_appear = False

    #MQTT Publish image limit
    count = 0

    #LoRa module connection
    lora_a = LoRaapi()

    #LED flash
    t = threading.Thread(target=flash,name='flash')
    t.start()

    next = True

    while next==True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image, appear_bird, bird_count, day, day_time = yolo.detect_image(image)
        
        if appear_bird==True:
            next = False
            print("Open U.S !!")
            lora_a.open("01")
            time.sleep(10)
            lora_a.close("01")
            time.sleep(15)
            next = True
        """
        if appear_bird==True and second_appear==True:
            print("Open U.S !!")
            lora_a.open("01")
            #time.sleep(2)
            #lora_a.open("02")
            time.sleep(10)
            lora_a.close("01")
            #time.sleep(2)
            #lora_a.close("02")
            time.sleep(15)
            appear_bird = False
            second_appear = False
        elif appear_bird==True:
            second_appear = True
        elif appear_bird==False:
            second_appear = False
        """

        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #cv2.imshow("result", result)
        
        if bird_count>0:
            result = cv2.resize(result, (640,360))
            cv2.imwrite("/home/jetson/Desktop/keras-yolo3_net/result.jpg", result)
            send_data(bird_count, day, day_time)
            #MQTT publish messenges 

        #if isOutput:
            #out.write(result)
    
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    frame.release()
    yolo.close_session()
