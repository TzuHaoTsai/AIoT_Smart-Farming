# AIoT_Smart-Farming

## Abstract
In recent years, with the maturity of deep learning technology, the problem of image recognition has become easier to solve, and it has begun to be applied to various industries. This paper attempts to apply this technology to solve the problem of bird damage in traditional agriculture. Among agricultural damages, apart from natural disasters, the largest case should be the eating disasters of birds or insects. In this paper, we propose the architecture and prototype design of a bird repellent system based on artificial intelligence of things (AIoT). In this system, we train a neural network model that can recognize birds, and implement it on a small single-board computer equipped with GPU (NVIDIA Jetson Nano). The system will be set up in the farmland, and capture the image via camera. If there is a bird group, it will activate the ultrasonic bird-repellent device through a long-range LPWAN  (Low-Power WAN) wireless network  to drive the bird away.

## System structure
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/System%20structure.png)

![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/%E6%99%BA%E6%85%A7%E9%B3%A5%E7%BE%A4%E8%BE%A8%E8%AD%98%E6%94%9D%E5%BD%B1%E6%A9%9F.png)
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/%E8%B6%85%E9%9F%B3%E6%B3%A2%E9%A9%85%E9%B3%A5%E5%99%A8.png)

## 實驗步驟

### 1.1 MediaTek LinkIt 7697
請將 LinkIt 7697 code/US_Launcher.ino 燒入於 LinkIt 7697 中，並且完成 LinkIt 7967 與超音波發射器的電路連接。

### 1.2 NVIDIA Jetson Nano
請參考[1]說明書來完成系統的架設與安裝，隨後至[2]安裝 tensorflow 等相依套件，完成後即可進入下一個步驟。

*本文作者採用 Jetpack SDK 4.2 的映像檔於 Jetson Nano，系統為 Ubuntu(18.04版) 的作業系統，含 CUDA(10.0版) 與 python(3.6.9版)。

### 1.3 YOLO: Real-Time Object Detection
本文作者採用 Ubuntu 作業系統來架設 YOLOv3 的模型訓練環境，請參考[3]說明書來架設運行環境，就可以開始進行模型的訓練。

### 1.4 Model Training
本文作者從[4]COCO dataset(2014)取得2240張含有鳥類的圖片(images)與其標記(Annotations)檔，並捨去一些在農田中較少出現的鳥類圖片，保留了810張圖片當作訓練用的資料集(https://reurl.cc/MZvRzW) ，可參考文章(https://reurl.cc/WEdDzZ) 來學習如何建立資料集與訓練參數之修改，最後運行[3]所提供的模型訓練程式碼，即可得到訓練完成的神經網路模型。

您可以運行[3]所提供的影像推論程式碼來評估模型的好壞，若準確率為佳，就可以在 NVIDIA Jetson Nano 上採用 keras-yolo3(https://github.com/qqwweee/keras-yolo3) 進行影像推論。

*而您也可以蒐集更多具有鳥群物件的圖片，並自行對圖片進行標記，就可以訓練出屬於自己的神經網路模型。

### 1.5 Inference
在 NVIDIA Jetson Nano 上採用 keras-yolo3 進行影像推論前，需要做模型的型態轉換，運行 keras-yolo3 所提供的 convert.py 程式碼，即可將 YOLOV3 模型檔轉換成適用於 Keras 框架的模型檔。

本文是採用畫面寬高值為 1280(px)*720(px)、AVC視訊編碼方式的測試影片(https://reurl.cc/4yRQbj) 作為驗證使用，而我們運行 keras-yolo3 所提供的影像推論程式(yolo_video.py)，由 OpenCV 套件導入測試用的影片(test.mp4)，且使用 keras 套件導入先前轉換完的模型檔，即可依序地對每一張圖片進行影像推論，而實驗過程中觀察到影像串流的 fps(Frame per Second)落在4~6幀。

*運行影像推論程式(yolo_video.py)時所需的主要套件為 keras(2.2.4版)、tensorflow-gpu(1.13.1版)。

![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/result.jpg)

*您也可以採用像是 Caffe、TensorRT 等模型框架進行影像推論，就可能達到更快的推論速度。

### 1.6	MQTT Broker
若您想使用攝影鏡頭讀取影像進行影像推論，並且將 NVIDIA Jetson Nano 的資料上傳至雲端，就得要把 keras-yolo3 的 yolo.py 捨棄掉，取代為本文的 Jetson Nano code/yolo.py，我們添加了 LoRa module 指令傳送、MQTT 資料上雲等程式碼。

從官方網站下載[5]mosquitto軟體並開啟此服務，在本機 Window 10 作業系統中點選”開始” → 搜尋”電腦管理” → 滑鼠左鍵點選 ”服務與應用程式” → 滑鼠左鍵點選 ”服務” → 滑鼠右鍵點選 ”Mosquitto Broker” ，即可啟動服務。

下載[6]MQTT.fx 軟體來測試 MQTT Broker 是否能成功運行，預先設定好 IP Address 與 port 並成功連線，再測試對主題(Topic)的發布與訂閱(Publish/Subscribe)。

### 1.7	Node-RED & MariaDB 
安裝[7]MariaDB 資料庫管理系統，隨後開啟 HeidiSQL 資料庫管理工具，即可新增欲建立的資料庫網路類型。

*以下圖片是作者所建立的資料庫
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/SQL_1.png)

成功創建資料庫後，即可新增資料表與資料欄位。

*以下圖片是作者所新增的資料表與資料欄位
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/SQL_2.png)

下載[8] Node.js 與 Node-RED，並導入 Server code/MQTT_DBS_Chart.json。

*請點擊紅色框選處的節點，並且將伺服端欄位設定為您架設 MQTT Broker 的 IP Address 與 Port
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/images/NodeRED_1.png)

經過部署後，Node-RED 就可撈取 MQTT Broker 中特定主題(Topic)的資料，並且將資料新增至資料庫。

### 1.8 Server
本文作者採用的[9]PyCharm 作為後臺運作的環境，您需要創立一個新的專案，並且運行 Server code/read_mqtt_to_local，其主要功能為與 MQTT Broker 建立連線，並且將圖片儲存在本地端，運行 Server code/image_post.py 程式碼後，就可以將辨識出鳥群的圖片傳送到 Node-RED dashboard 上。

## Reference

[1] NVIDIA Jetson Nano - Preparation : https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

[2] NVIDIA Jetson Nano - Tensorflow-gpu : https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

[3] YOLO: Real-Time Object Detection : https://pjreddie.com/darknet/yolo/

[4]	COCO Dataset : https://cocodataset.org/#home

[5] Mosquitto : https://mosquitto.org/

[6] MQTT.fx : https://mqttfx.jensd.de/

[7] Mariadb : https://downloads.mariadb.org/mariadb/

[8] Node-RED & Node.js : https://nodered.org/docs/getting-started/windows#1-install-nodejs

[9] PyCharm : https://www.jetbrains.com/pycharm/

