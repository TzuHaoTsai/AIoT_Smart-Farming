import requests

count = 1

while True:
    temp = 'C:/Users/IoT/PycharmProjects/face/image/' + str(count) + ".jpg"

    files = {'image': open(temp, 'rb')}
    requests.post('http://localhost:1880/IMAGE', files = files)
    count = count + 1