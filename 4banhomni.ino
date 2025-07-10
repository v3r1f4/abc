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

float vx = 1;
float vy = 0;
float omega = 0;

BLA::Matrix<3,1> P = {vx, vy, omega};
BLA::Matrix<4,1> V = A_inv * P;

float dir[4] = {V(0,0), V(1,0), V(2,0), V(3,0)};

void setup() {
  for (size_t j = 0; j < 4; j++) {
    steppers[j] = new AccelStepper(HALFSTEP, output[j][0], output[j][2], output[j][1], output[j][3]);
    steppers[j]->setMaxSpeed(1000);
    steppers[j]->setAcceleration(motorAccel[j]);
  }
}

void loop() {
  float vx_arr[4] = {0, -1, 0, 1};
  float vy_arr[4] = {1, 0, -1, 0};
  float omega_arr[4] = {0, 0, 0, 0};

  // for (size_t i = 0; i < 4; i++) {
    // Cập nhật vector chuyển động
  P(0,0) = 1;
  P(1,0) = 0;
  P(2,0) = 0;

  V = A_inv * P;

  for (size_t j = 0; j < 4; j++) {
    dir[j] = V(j,0);
    steppers[j]->setSpeed(motorSpeed[j] * dir[j] * 1.45);
  }

  unsigned long start = millis();
  while (millis() - start < 10000) {
    for (size_t j = 0; j < 4; j++) {
      steppers[j]->runSpeed();
    }
  }
}