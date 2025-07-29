// MAC Address: 14:33:5c:04:61:18

#include <WiFi.h>
#include <esp_now.h>

typedef struct data_struct {
  int id;
  float vx;
  float vy;
  float omega;
} data_struct;

data_struct incomingData;

void OnDataRecv(const esp_now_recv_info *recv_info, const uint8_t *incomingDataRaw, int len) {
  memcpy(&incomingData, incomingDataRaw, sizeof(incomingData));
  Serial.print("Received id (int): ");
  Serial.println(incomingData.id);
  Serial.print("Received vx (float): ");
  Serial.println(incomingData.vx);
  Serial.print("Received vy (float): ");
  Serial.println(incomingData.vy);
  Serial.print("Received omega (float): ");
  Serial.println(incomingData.omega);
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {}