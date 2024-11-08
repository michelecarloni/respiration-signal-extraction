import numpy as np
import pyrealsense2 as rs
import asyncio
import time
import cv2
import math
from bleak import BleakClient
from bleak.uuids import uuid16_dict
from ultralytics import YOLO
import logging
import winsound

logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Tempo di esecuzione
EXECUTION_TIME = 120
# Frequenza telecamera Intel
INTEL_FREQUENCY = 30
# Risoluzione telecamera Intel
W = 640
H = 480

# MAC adress del polarH10
ADDRESS = "F8:B1:5D:44:0E:8B"

ECG_SAMPLING_FREQ = 130  # Frequenza di campionamento ECG
ACC_SAMPLING_FREQ = 200  # Frequenza di campionamento accelerazione (200 Hz)

# Definizioni costanti UUID
uuid16_dict = {v: k for k, v in uuid16_dict.items()}
MODEL_NBR_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Model Number String")
)

MANUFACTURER_NAME_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Manufacturer Name String")
)

BATTERY_LEVEL_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Battery Level")
)

PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])
ACC_WRITE = bytearray([0x02, 0x02, 0x00, 0x01, 0xC8, 0x00, 0x01, 0x01, 0x10, 0x00, 0x02, 0x01, 0x08, 0x00])

# Inizializzazione del modello YOLO
model_directory = r'C:\dev\intel RealSense\Algorithms YOLO\yolov8n-pose.pt'
model = YOLO(model_directory)

ecg_session_data = []
ecg_session_time = []
acc_session_data = []
acc_session_time = []

async def acquire_polar_data(client,st_time):
    try:
        
        ecg_session_data = []
        ecg_session_time = []
        acc_session_data = []
        acc_session_time = []
        
        start_time = time.time()
        print("Start Polar effettivo: ", start_time, " passato: ",st_time)
        winsound.Beep(1000,1000)
        

        def ecg_data_conv(sender, data):
            print("ENTRATO",time.time())
            if data[0] == 0x00:
                timestamp = convert_to_unsigned_long(data, 1, 8)
                step = 3
                samples = data[10:]
                offset = 0
                while offset < len(samples):
                    ecg = convert_array_to_signed_int(samples, offset, step)
                    offset += step
                    ecg_session_data.append(ecg)
                    ecg_session_time.append(timestamp)

        def acc_data_conv(sender, data):
            if data[0] == 0x02:
                timestamp = convert_to_unsigned_long(data, 1, 8)
                frame_type = data[9]
                resolution = (frame_type + 1) * 8  # Risoluzione in bit
                time_step = 1 / ACC_SAMPLING_FREQ  # Passo temporale tra i campioni
                step = math.ceil(resolution / 8.0)  # Numero di byte per campione
                samples = data[10:]
                n_samples = len(samples) // (step * 3)  # Numero di campioni presenti nei dati
                sample_timestamp = timestamp - (n_samples - 1) * time_step
                offset = 0
                while offset < len(samples):
                    x = convert_array_to_signed_int(samples, offset, step)
                    offset += step
                    y = convert_array_to_signed_int(samples, offset, step)
                    offset += step
                    z = convert_array_to_signed_int(samples, offset, step)
                    offset += step
                    acc_session_data.append([x, y, z])
                    acc_session_time.append(sample_timestamp)
                    sample_timestamp += time_step

        def convert_array_to_signed_int(data, offset, length):
            return int.from_bytes(
                bytearray(data[offset: offset + length]), byteorder="little", signed=True
            )

        def convert_to_unsigned_long(data, offset, length):
            return int.from_bytes(
                bytearray(data[offset: offset + length]), byteorder="little", signed=False,
            )

        await client.write_gatt_char(PMD_CONTROL, ECG_WRITE)
        await client.start_notify(PMD_DATA, ecg_data_conv)

        await client.write_gatt_char(PMD_CONTROL, ACC_WRITE)
        await client.start_notify(PMD_DATA, acc_data_conv)

        # Attendere un certo periodo di tempo per la registrazione dei dati
        await asyncio.sleep(EXECUTION_TIME)
        print("finish polar: ", time.time())
        winsound.Beep(1000,1000)
        
        # if time.time() - start_time >= 20:
        #     print("finish polar: ", time.time())
        return ecg_session_time, ecg_session_data, acc_session_time, acc_session_data

    except Exception as e:
        print("Errore durante la connessione a Polar:", e)
    finally:
        await client.disconnect()

async def acquire_realsense_data(config, st_time, align):
    pipeline = rs.pipeline()
    pipeline.start(config)

    color_data = []
    depth_data = []
    timestamps = []

    start_time = time.time()
    print("start Intel: ", start_time," passato: ",st_time)
    
    num_frames = 0
    while True:
        timestamp = time.time() - start_time
        if timestamp >= EXECUTION_TIME:
            break

        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
             print("frame perso")
             continue

        # if not frames:
        #     print("frame perso")
        #     continue

        #color_frame = frames.get_color_frame()
        #depth_frame = frames.get_depth_frame()

        #if not color_frame or not depth_frame:
        #    print("colori persi")
        #    continue
        color_data.append(np.array(color_frame.get_data()))
        depth_data.append(np.array(aligned_depth_frame.get_data()))
                
        timestamps.append(time.time() - start_time)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Display color and depth image
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)
        cv2.waitKey(1)

        num_frames += 1

        await asyncio.sleep(1.0 / INTEL_FREQUENCY)  # Frequenza di campionamento dei frame

    pipeline.stop()

    print("Finish Intel: ", time.time())

    return color_data, depth_data, timestamps


async def connect_to_polar(client):
    try:
        await client.connect()
        await client.is_connected()
        print("---------Device connected--------------")

        model_number = await client.read_gatt_char(MODEL_NBR_UUID)
        manufacturer_name = await client.read_gatt_char(MANUFACTURER_NAME_UUID)
        battery_level = await client.read_gatt_char(BATTERY_LEVEL_UUID)

        print("Model Number: {0}".format("".join(map(chr, model_number))))
        print("Manufacturer Name: {0}".format("".join(map(chr, manufacturer_name))))
        print("Battery Level: {0}%".format(int(battery_level[0])))

        return True
    except Exception as e:
        print("Errore durante la connessione a Polar:", e)
        return False


async def main():
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, INTEL_FREQUENCY)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, INTEL_FREQUENCY)
    
    align_to = rs.stream.color
    align = rs.align(align_to)

    client = BleakClient(ADDRESS)
    connected = await connect_to_polar(client)

    if connected:
        st_time=time.time()
        result_intel,result_polar = await asyncio.gather(acquire_realsense_data(config,st_time, align), acquire_polar_data(client,st_time))
        print("End Intel | End Polar processing")

        print("start Polar savings: ", time.time())

        ecg_session_time = np.array(result_polar[0])
        ecg_session_data = np.array(result_polar[1])
        acc_session_time = np.array(result_polar[2])
        acc_session_data = np.array(result_polar[3])

        np.save("polar_ecg_time.npy", ecg_session_time)
        np.save("polar_ecg_data.npy", ecg_session_data)
        np.save("polar_acc_time.npy", acc_session_time)
        np.save("polar_acc_data.npy", acc_session_data)

        print("End Polar savings: ", time.time())

        print("start Intel savings: ", time.time())

        color_data = np.array(result_intel[0])
        depth_data = np.array(result_intel[1])
        timestamps_intel = np.array(result_intel[2])

        np.save("color_data.npy", color_data)
        np.save("depth_data.npy", depth_data)
        np.save("timestamps_realsense.npy", timestamps_intel)
        print(timestamps_intel)

        print("End Intel savings: ", time.time())
        
        
        print(f"len color: {len(color_data)}")
        print(f"len depth: {len(depth_data)}")

        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())