#include <AccelStepper.h>
#include <BasicLinearAlgebra.h>
using namespace BLA;

const size_t output[4][4] = {
  {24, 25, 26, 27},
  {28, 29, 30, 31},
  {32, 33, 34, 35},
  {36, 37, 38, 39}
};

#define HALFSTEP 8

AccelStepper* steppers[4];

size_t motorSpeed[4] = {900, 900, 900, 900}; // 500 - 900
size_t motorAccel[4] = {200, 200, 200, 200};  
bool isRunning[4] = {true, true, true, true};

BLA::Matrix<4,3> A_inv = {
  -0.35355, -0.35355, -3.33333,
  -0.35355, 0.35355, -3.33333,
  0.35355, 0.35355, 3.33333,
  0.35355, -0.35355, 3.33333
};

float vx = 0;
float vy = 0;
float omega = 0;

BLA::Matrix<3,1> P = {vx, vy, omega};
BLA::Matrix<4,1> V = A_inv * P;

float x_rb = 0;
float y_rb = 0;
float theta = 0;

float x1 = 1;
float y1 = 3;

int v = 15;

float Kp = 2.5;
float Ki = 0.05;
float Kd = 0.3;

float error_prev = 0;
float error_integral = 0;
float distance_threshold = 0.01;

void setup() {
  for (size_t j = 0; j < 4; j++) {
    steppers[j] = new AccelStepper(HALFSTEP, output[j][0], output[j][2], output[j][1], output[j][3]);
    steppers[j]->setMaxSpeed(1000);
    steppers[j]->setAcceleration(motorAccel[j]);
  }

  Serial1.begin(38400);
}

void loop() {
  static unsigned long last_time = 0;
  unsigned long current_time = millis();
  float dt = (current_time - last_time) / 1000.0;
  if (last_time == 0) dt = 0.02;
  last_time = current_time;
  
  float distance = sqrt(sq(x1 - x_rb) + sq(y1 - y_rb));

  if (distance < distance_threshold) { // Dieu kien dung
    P(0,0) = 0;
    P(1,0) = 0;
    P(2,0) = 0;

    for (size_t j = 0; j < 4; j++) {
      steppers[j]->setSpeed(0);
    }
  }

  float beta = atan2(y1-y_rb, x1-x_rb);

  float error = atan2(sin(beta - theta), cos(beta - theta));

  float error_integral = error_integral + error * dt;
  float error_derivative = (error - error_prev) / dt;
  float omega = Kp * error + Ki * error_integral + Kd * error_derivative;

  float v_current = min(v, max(5, distance * 2));

  float vx = v_current * cos(theta);
  float vy = v_current * sin(theta);

  x_rb += vx * dt;
  y_rb += vy * dt;
  theta += omega * dt;

  theta = atan2(sin(theta), cos(theta));

  P(0,0) = vx;
  P(1,0) = vy;
  P(2,0) = omega;

  V = A_inv * P;

  error_prev = error;

  float vx_arr[4] = {0, -1, 0, 1};
  float vy_arr[4] = {1, 0, -1, 0};
  float omega_arr[4] = {0, 0, 0, 0};

  for (size_t j = 0; j < 4; j++) {
    float dir = V(j,0);
    steppers[j]->setSpeed(motorSpeed[j] * dir * 1.45);
  }

  unsigned long start = millis();
  while (millis() - start < 20) {
    for (size_t j = 0; j < 4; j++) {
      steppers[j]->runSpeed();
    }
  }

  if (Serial1.available()) {
    String receivedData = Serial1.readStringUntil('\n');
    receivedData.trim();
    
    Serial.println("Received from ESP32: " + receivedData);
    
    processCommand(receivedData);
  }
}

void processCommand(String command) {
  int firstSpace = command.indexOf(' ');
  int secondSpace = command.indexOf(' ');

  float vx = command.substring(0, firstSpace).toFloat();
  float vy = command.substring(firstSpace + 1, secondSpace).toFloat();
  float omega = command.substring(secondSpace + 1).toFloat();

  Serial.printf("Parsed: vx=%.2f, vy=%.2f, omega=%.2f\n", vx, vy, omega);
}
