#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <AccelStepper.h>

// =====================================================
// PCA9685
// =====================================================
Adafruit_PWMServoDriver pca9685 =
    Adafruit_PWMServoDriver(0x40);

// =====================================================
// SERVO CHANNELS
// =====================================================
#define SHOULDER_CH 0
#define ELBOW_CH    2
#define WRIST_CH    3

// =====================================================
// SERVO PWM LIMITS
// =====================================================
#define SERVO_MIN 110
#define SERVO_MAX 510

// =====================================================
// SHOULDER OFFSET
// =====================================================
#define SHOULDER_OFFSET 30
#define WRIST_OFFSET -20
// =====================================================
// CNC SHIELD Z AXIS
// =====================================================
#define Z_STEP_PIN 4
#define Z_DIR_PIN  7

// =====================================================
// ACCELSTEPPER
// =====================================================
AccelStepper railStepper(
    AccelStepper::DRIVER,
    Z_STEP_PIN,
    Z_DIR_PIN
);

// =====================================================
// RAIL SETTINGS
// =====================================================

// 1/8 microstepping
// GT2 20T pulley
float stepsPerMM = 150;

// Rail limits
float railMinMM = 0;
float railMaxMM = 300;

// Stable tested settings
float railMaxSpeed = 7000;

float railAcceleration = 3000;

// =====================================================
// ARM SETTINGS
// =====================================================
float armMaxStep = 1.2;

// =====================================================
// CURRENT POSITIONS
// =====================================================
float shoulderPos = 90;
float elbowPos = 90;
float wristPos = 90;

// =====================================================
// TARGET POSITIONS
// =====================================================
float shoulderTarget = 90;
float elbowTarget = 90;
float wristTarget = 90;

float railTargetMM = 0;

// =====================================================
// ANGLE -> PWM
// =====================================================
int angleToPulse(float angle)
{
    angle = constrain(angle, 0, 180);

    return map(
        angle,
        0,
        180,
        SERVO_MIN,
        SERVO_MAX
    );
}

// =====================================================
// ELBOW REVERSE
//
// 0 -> 180
// 180 -> 0
// =====================================================
float mapElbow(float angle)
{
    return 180.0 - angle;
}

// =====================================================
// WRITE SERVO
// =====================================================
void writeServo(uint8_t ch, float angle)
{
    angle = constrain(angle, 0, 180);

    int pulse = angleToPulse(angle);

    pca9685.setPWM(ch, 0, pulse);
}

// =====================================================
// SMOOTH JOINT MOTION
// =====================================================
void moveJoint(float &current, float target)
{
    float diff = target - current;

    // deadband
    if (abs(diff) < 1.0)
        return;

    // speed limit
    if (diff > armMaxStep)
        diff = armMaxStep;

    if (diff < -armMaxStep)
        diff = -armMaxStep;

    current += diff;
}

// =====================================================
// ARM REACHED TARGET?
// =====================================================
bool armReached()
{
    if (abs(shoulderPos - shoulderTarget) > 2)
        return false;

    if (abs(elbowPos - elbowTarget) > 2)
        return false;

    if (abs(wristPos - wristTarget) > 2)
        return false;

    return true;
}

// =====================================================
// SERIAL FORMAT
//
// R150,S90,E130,W40
//
// R = rail mm
// =====================================================
void readSerial()
{
    if (Serial.available())
    {
        String line =
            Serial.readStringUntil('\n');

        line.trim();

        int rIndex = line.indexOf('R');
        int sIndex = line.indexOf('S');
        int eIndex = line.indexOf('E');
        int wIndex = line.indexOf('W');

        if (
            rIndex == -1 ||
            sIndex == -1 ||
            eIndex == -1 ||
            wIndex == -1
        )
        {
            return;
        }

        int comma1 =
            line.indexOf(',', rIndex);

        int comma2 =
            line.indexOf(',', sIndex);

        int comma3 =
            line.indexOf(',', eIndex);

        String rVal =
            line.substring(rIndex + 1, comma1);

        String sVal =
            line.substring(sIndex + 1, comma2);

        String eVal =
            line.substring(eIndex + 1, comma3);

        String wVal =
            line.substring(wIndex + 1);

        railTargetMM =
            constrain(
                rVal.toFloat(),
                railMinMM,
                railMaxMM
            );

        shoulderTarget =
            constrain(
                sVal.toFloat(),
                0,
                180
            );

        elbowTarget =
            constrain(
                eVal.toFloat(),
                0,
                180
            );

        wristTarget =
            constrain(
                wVal.toFloat(),
                0,
                180
            );
    }
}

// =====================================================
// SETUP
// =====================================================
void setup()
{
    Serial.begin(115200);

    // ================= PCA9685 =================

    pca9685.begin();

    pca9685.setPWMFreq(50);

    // ================= RAIL =================

    railStepper.setMinPulseWidth(20);

    railStepper.setMaxSpeed(
        railMaxSpeed
    );

    railStepper.setAcceleration(
        railAcceleration
    );

    delay(1000);

    // ================= INITIAL ARM =================

    writeServo(
        SHOULDER_CH,
        shoulderPos + SHOULDER_OFFSET
    );

    writeServo(
        ELBOW_CH,
        mapElbow(elbowPos)
    );

    writeServo(
        WRIST_CH,
        wristPos
    );

    Serial.println("READY");
}

// =====================================================
// LOOP
// =====================================================
void loop()
{
    // ================= READ SERIAL =================

    readSerial();

    // ================= ARM =================

    moveJoint(
        shoulderPos,
        shoulderTarget
    );

    moveJoint(
        elbowPos,
        elbowTarget
    );

    moveJoint(
        wristPos,
        wristTarget
    );

    // Shoulder offset
    float shoulderMapped =
        shoulderPos + SHOULDER_OFFSET;

    // Elbow reverse
    float elbowMapped =
        mapElbow(elbowPos);

    // Write servos
    writeServo(
        SHOULDER_CH,
        shoulderMapped
    );

    writeServo(
        ELBOW_CH,
        elbowMapped
    );

float wristMapped =
    wristPos + WRIST_OFFSET;

writeServo(
    WRIST_CH,
    wristMapped
);

    // ================= RAIL =================
    // Rail moves ONLY after arm settles

    if (armReached())
    {
        long targetSteps =
            railTargetMM * stepsPerMM;

        railStepper.moveTo(targetSteps);
    }

    // Smooth non-blocking motion
    railStepper.run();
}