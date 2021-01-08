# AIoT_Smart-Farming
This system is in the development stage.

## System structure
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/Smart-Farming-System.png)

## 軟體架設

### 1.1	MQTT Broker 架設流程
從官方網站[7]下載mosquitto軟體並開啟此服務，Window 10 作業系統中點選”開始” → 搜尋”電腦管理” → 滑鼠左鍵點選 ”服務與應用程式” → 滑鼠左鍵點選 ”服務” → 滑鼠右鍵點選 ”Mosquitto Broker” ，即可啟動服務。
至[8]下載MQTT.fx軟體來測試MQTT Broker是否能成功運行，預先設定好IP Address與port並成功連線，再測試對主題(Topic)的發布與訂閱(Publish/Subscribe)。

### 1.2	Node-RED 與 MariaDB 的架設與運作流程
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

[7] Mosquitto 軟體安裝 : https://mosquitto.org/

[8] MQTT.fx 軟體安裝 : https://mqttfx.jensd.de/

[9] Mariadb 軟體安裝 : https://downloads.mariadb.org/mariadb/

[10] Node-RED & Node.js 軟體安裝 : https://nodered.org/docs/getting-started/windows#1-install-nodejs



