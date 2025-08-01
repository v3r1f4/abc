const int IN1 = 5;
const int IN2 = 4;
const int IN3 = 6;
const int IN4 = 7;

float target[100][2];
int i = 0;

void dk_dc(int pwmL, int pwmR) {
  analogWrite(IN1, pwmL > 0 ? pwmL : 0);
  analogWrite(IN2, pwmL < 0 ? -pwmL : 0);
  analogWrite(IN3, pwmR > 0 ? pwmR : 0);
  analogWrite(IN4, pwmR < 0 ? -pwmR : 0);
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);  // RX1 = pin 19
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
}

void loop() {
  if (Serial1.available()) {
    String msg = Serial1.readStringUntil('\n');
    msg.trim();

    Serial.print("Mega nhận: '");
    Serial.print(msg);
    Serial.print("' (length: ");
    Serial.print(msg.length());
    Serial.println(")");

    float values[3];
    int n = 0;
    int startIndex = 0;

    // xử lý tín hiệu nhận
    for (int j = 0; j <= msg.length(); j++) {
      if (j == msg.length() || msg[j] == ' ') {
        if (j > startIndex) {
          String numberStr = msg.substring(startIndex, j);
          values[n] = numberStr.toFloat();
          n++;
          if (n >= 3) break;
        }
        startIndex = j + 1;
      }
    }

    Serial.print("n = ");
    Serial.println(n);

    if (n == 3) {  // gửi vị trí
      float x = values[0];
      float y = values[1];
      float omega = values[2];
      Serial.println("Vị trí hiện tại của robot:");
      Serial.print("x = ");
      Serial.print(x);
      Serial.print(", y = ");
      Serial.print(y);
      Serial.print(", omega = ");
      Serial.println(omega);
    } else if (n == 2) {  // thêm vị trí đích
      target[i][0] = values[0];
      target[i][1] = values[1];
      Serial.println("n == 2 - Vị trí đích mới:");
      for (int j = 0; j <= i; j++) {
        Serial.print("x["); Serial.print(j); Serial.print("] = ");
        Serial.print(target[j][0]);
        Serial.print(", y["); Serial.print(j); Serial.print("] = ");
        Serial.println(target[j][1]);
      }
      i += 1;
    }
  }
  delay(1000);
}