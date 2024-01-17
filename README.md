# CameraIMU-calibration
Paper: Camera-IMU Extrinsic Calibration Method Based on Intermittent Sampling and RANSAC Optimization

The approach entails detecting motion-to-static transition points in the IMU data as keyframe segmentation criteria, thereby decoupling data acquisition from system timestamps and aligning it with the sensor's motion process. Subsequently, the robot hand-eye calibration method is applied to the extrinsic calibration of the Camera-IMU, combining the Random Sample Consensus (RANSAC) algorithm to filter the poses of the camera and IMU. Through this process, precise calibration of the extrinsic parameters is achieved.

### Download Code and Dataset

1. Clone the Pot_Detection repository
    ```Shell
    https://github.com/LiangXinfeng/Camera-IMU-calibration
    ```
    
2. Main.py is used for camera and IMU data sampling. The sampled files are stored in the ‘DATA’ folder. The sampled images are stored in the ‘PHOTO’ folder.

3. DataFiltering.py is used to filter images stored in the ‘PHOTO’ folder and match IMU and camera data. The updated results are stored in the ‘DATA’ folder.

4. Calibr_ Data.py is used to integrate data from multiple experiments.

5. AutomaticHand Eye. py is used for external parameter calibration of integrated data


   

