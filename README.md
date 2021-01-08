# AIoT_Smart-Farming
This system is in the development stage.

## System structure
![image](https://github.com/TzuHaoTsai/AIoT_Smart-Farming/blob/main/Smart-Farming-System.png)

## 軟體架設

### 1.1	MQTT Broker 架設
架設MQTT Broker伺服器須從官方網站下載mosquitto軟體，如[7]，下載後須要開啟服務，從點選”開始” → 搜尋”電腦管理” → 滑鼠左鍵點選 ”服務與應用程式” → 滑鼠左鍵點選 ”服務” → 滑鼠右鍵點選 ”Mosquitto Broker” 以啟動服務。
如[8]，下載MQTT.fx軟體來測試MQTT Bro-ker是否能成功運行，先設定好IP Address 與 port並成功連線，再測試對Topic的Publish/Subscribe。

### 1.2	Node-RED 與Database之應用
至MariaDB官網如[9]下載與安裝軟體，同時會將HeidiSQL一並安裝，HeidiSQL是資料庫GUI管理工具，方便使用者操作資料庫，我們開啟HeidiSQL並連接MariaDB，然後創建資料表與資料欄位。
再到Node.js與Node-RED的官網如[10]，下載應用軟體，Node-RED負責與MQTT Broker進行連線，使用Node-RED訂閱指定的Topic且將資料撈取回來，最後將資料新增到資料庫中。

## Tools

#### Model Training tools
https://pjreddie.com/darknet/yolo/

#### Training set & Testing set
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

#### Model Inference
https://github.com/qqwweee/keras-yolo3

#### Our Dataset , annotation and trained model
https://reurl.cc/ygDgmy

[7] Mosquitto 軟體安裝 : https://mosquitto.org/

[8] MQTT.fx 軟體安裝 : https://mqttfx.jensd.de/

[9] Mariadb 軟體安裝 : https://downloads.mariadb.org/mariadb/

[10] Node-RED & Node.js 軟體安裝 : https://nodered.org/docs/getting-started/windows#1-install-nodejs


