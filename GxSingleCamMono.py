# version:1.0.1905.9051
import gxipy as gx
from PIL import Image
import main
import cv2
numpy_image = 1
def CamMono(image_array):
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    cam = device_manager.open_device_by_index(1)
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    cam.ExposureTime.set(25000)
    cam.Gain.set(10.0)
    cam.stream_on()
    num = 10000
    for i in range(num):
        # get raw image
        raw_image = cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed.")
            continue
        # create numpy array with data from raw image
        numpy_image = raw_image.get_numpy_array()
        numpy_image2 = cv2.resize(numpy_image, (1224, 1024))
        if i == 10:                  # 自动记录相机初始图像
            filename = "D:/CAMERA-IMU/PHOTO/1.jpg"
            cv2.imwrite(filename, numpy_image2)
        image_array[:] = numpy_image2
        if numpy_image is None:
            continue
        # show acquired image
        # img = Image.fromarray(numpy_image, 'L')

        # image_array.put(numpy_image)
        cv2.imshow('input', numpy_image2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # main.program_exit()
            break
    cam.stream_off()
    cam.close_device()
