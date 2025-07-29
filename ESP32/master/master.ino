// MAC Address: d0:ef:76:47:4d:34

#include <esp_now.h>
#include <WiFi.h>

// REPLACE WITH YOUR RECEIVER MAC Address
uint8_t broadcastAddress1[] = {0x14, 0x33, 0x5C, 0x04, 0x61, 0x18};
uint8_t broadcastAddress2[] = {0x1C, 0x69, 0x20, 0xA4, 0xD0, 0x58};
uint8_t broadcastAddress3[] = {0x94, 0xB9, 0x7E, 0xFB, 0x23, 0xF0};

// Structure example to send data
// Must match the receiver structure
typedef struct data_struct {
  int id;
  float vx;
  float vy;
  float omega;
} data_struct;

// Create a struct_message called myData
data_struct myData;

esp_now_peer_info_t peerInfo;

// callback when data is sent
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("\r\nLast Packet Send Status:\t");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}
 
void setup() {
  // Init Serial Monitor
  Serial.begin(115200);
 
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Once ESPNow is successfully Init, we will register for Send CB to
  // get the status of Trasnmitted packet
  esp_now_register_send_cb(OnDataSent);
  
  // Register peer
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;

  // Regster first peer
  memcpy(peerInfo.peer_addr, broadcastAddress1, 6);    
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }

  // Regster second peer
  memcpy(peerInfo.peer_addr, broadcastAddress2, 6);    
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }

  // Regster third peer
  memcpy(peerInfo.peer_addr, broadcastAddress3, 6);    
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }
}
 
void loop() {
  // Set values to send
  myData.id = 1;
  myData.vx = 2.3;
  myData.vy = 2.2;
  myData.omega = 3.3;
  
  // Send message via ESP-NOW
  esp_err_t result = esp_now_send(broadcastAddress1, (uint8_t *) &myData, sizeof(myData));
   
  if (result == ESP_OK) {
    Serial.println("Sent with success");
  }
  else {
    Serial.println("Error sending the data");
  }
  delay(2000);
}