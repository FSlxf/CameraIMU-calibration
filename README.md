# Camera-IMU-calibration
Paper: Camera-IMU Extrinsic Calibration Method Based on Intermittent Sampling and RANSAC Optimization

The approach entails detecting motion-to-static transition points in the IMU data as keyframe segmentation criteria, thereby decoupling data acquisition from system timestamps and aligning it with the sensor's motion process. Subsequently, the robot hand-eye calibration method is applied to the extrinsic calibration of the Camera-IMU, combining the Random Sample Consensus (RANSAC) algorithm to filter the poses of the camera and IMU. Through this process, precise calibration of the extrinsic parameters is achieved.

### Download Code and Dataset

1. Clone the Pot_Detection repository
    ```Shell
    git clone https://github.com/SunCihan/Pot_detection.git
    ```
