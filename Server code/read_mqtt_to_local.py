import paho.mqtt.client as mqtt
MQTT_SERVER = "127.0.0.1"
MQTT_TOPIC = "AIoT/bird_images"

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_TOPIC)
    # The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
    # more callbacks, etc
    # Create a file with write byte permission
    f = open("output1.jpg", "wb")
    f.write(msg.payload)
    print("Image Received")
    f.close()

client = mqtt.Client()  #確保都有連線到
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_SERVER, 18883, 60)
client.loop_forever()