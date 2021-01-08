# AIoT_Smart-Farming
This system is in the development stage.

## System structure
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/Smart-Farming-System.png)

## 實驗步驟

### 1.1 NVIDIA Jetson Nano
請參考[1]說明書來完成系統的架設與安裝，隨後至[2]安裝tensorflow等相依套件，完成後即可進入下一個步驟。

*本文作者採用 Jetpack SDK 4.2的映像檔於Jetson Nano，系統為 Ubuntu(18.04版)的作業系統、CUDA(10.0版)，並且含有 python(3.6.9版)軟體，而執行影像推論程式碼(YOLOv3)所需的主要套件為keras(2.2.4版)、tensorflow-gpu(1.13.1版)。

### 1.2 YOLO: Real-Time Object Detection
本文作者採用Ubuntu作業系統來架設YOLOv3的模型訓練環境，請參考[3]說明書來架設運行環境，就可以開始進行鳥群物件辨識的模型訓練。

### 1.3 Model Training
本文作者採用COCO dataset(2014)[4]取得2240張含有鳥類的圖片(images)與其標記(Annotations)檔，並捨去一些在農田中較少出現的鳥類圖片，保留了810張圖片當作訓練用的資料集，接下來運行官網[3]提供的程式碼來進行模型的訓練，隨後將產生出來的模型檔進行影像推論，並且驗證其模型之成效。

*而您也可以蒐集更多具有鳥群物件的圖片，並自行對圖片進行標記，就可以訓練出屬於自己的神經網路模型。

### 1.4 Inference
經過模型的驗證後，若準確率為佳，即可在NVIDIA Jetson Nano上進行影像推論，使用OpenCV套件導入預錄好的影片，並且透過keras套件將訓練完成的模型導入程式中，依序地對每一張影像進行模型推論，而實驗過程中觀察到影像的fps(Frame per Second)落在4~6幀，

*您也可以採用其他模型套件進行影像推論，像是Caffe、。

### 1.	MQTT Broker
從官方網站[7]下載mosquitto軟體並開啟此服務，Window 10 作業系統中點選”開始” → 搜尋”電腦管理” → 滑鼠左鍵點選 ”服務與應用程式” → 滑鼠左鍵點選 ”服務” → 滑鼠右鍵點選 ”Mosquitto Broker” ，即可啟動服務。
至[8]下載MQTT.fx軟體來測試MQTT Broker是否能成功運行，預先設定好IP Address與port並成功連線，再測試對主題(Topic)的發布與訂閱(Publish/Subscribe)。

### 1.	Node-RED & MariaDB 
至[9]安裝MariaDB資料庫管理系統，安裝後即可開啟HeidiSQL資料庫管理工具，新增並選擇想要建立的網路類型，創建後即可創建資料表與填寫資料欄位。
再到[10]下載 Node.js與Node-RED，我們即可透過Node-RED撈取MQTT Broker中特定主題(Topic)的資料，並且將資料新增到資料庫中。

## Tools

#### Model Training tools
https://pjreddie.com/darknet/yolo/

#### Training set & Testing set
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

#### Model Inference
https://github.com/qqwweee/keras-yolo3

#### Our Dataset , annotation and trained model
https://reurl.cc/ygDgmy

## Reference

[1] NVIDIA Jetson Nano : https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

[2] NVIDIA Jetson Nano : https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

[3] YOLO: Real-Time Object Detection : https://pjreddie.com/darknet/yolo/

[4]	COCO Dataset : https://cocodataset.org/#home

[7] Mosquitto : https://mosquitto.org/

[8] MQTT.fx : https://mqttfx.jensd.de/

[9] Mariadb 軟體安裝 : https://downloads.mariadb.org/mariadb/

[10] Node-RED & Node.js 軟體安裝 : https://nodered.org/docs/getting-started/windows#1-install-nodejs



