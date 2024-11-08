import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
import winsound    

def cornice (shoulder1, shoulder2, pelvis1, pelvis2, depth_image):
    
    range = 0.1

    distanza_shoulder_shoulder = np.linalg.norm(shoulder2 - shoulder1)
    distanza_pelvis_pelvis = np.linalg.norm(pelvis2 - pelvis1)

    offsetS = int(range * distanza_shoulder_shoulder)
    offsetB = int(range * distanza_pelvis_pelvis)

    shoulder1_in = [shoulder1[0] - offsetS, shoulder1[1]]
    shoulder2_in = [shoulder2[0] + offsetS, shoulder2[1]]
    pelvis1_in = [pelvis1[0] - offsetB, pelvis1[1]]
    pelvis2_in = [pelvis2[0] + offsetB, pelvis2[1]]

    poly_pts_in = np.array([shoulder1_in, shoulder2_in, pelvis2_in, pelvis1_in], dtype=np.int32)
    poly_pts_out = np.array([shoulder1, shoulder2, pelvis2, pelvis1], dtype=np.int32)

    mask_in = np.zeros_like(depth_image, dtype=np.uint8)
    mask_out = np.zeros_like(depth_image, dtype=np.uint8)
    cv2.fillPoly(mask_in, [poly_pts_in], 255)
    cv2.fillPoly(mask_out, [poly_pts_out], 255)

    mask_edge = cv2.subtract(mask_out, mask_in)

   # cv2.imwrite('mask_in.png', mask_in)
   # cv2.imwrite('mask_out.png', mask_out)
    cv2.imwrite('mask_edge.png', mask_edge)

    depth_values = depth_image[mask_edge == 255]
    depth_values_in_centimeters = depth_values / 10.0

    print(f"Depth values: {depth_values}")
    print(f"Depth values in cm: {depth_values_in_centimeters}")

    mean_depth = np.mean(depth_values_in_centimeters) if len(depth_values_in_centimeters) > 0 else 0

    return mean_depth

def torsoPiccolo (shoulder1, shoulder2, pelvis1, pelvis2, depth_image):
    
    shoulder1_in = shoulder1    
    shoulder2_in = shoulder2  
    pelvis1_in = pelvis1   
    pelvis2_in = pelvis2    
    
    rangeSX = 0.2
    rangeSY = 0.2
    rangeBX = 0.2
    rangeBY = 0.6
    
    
    distanza_shoulder_shoulder = np.linalg.norm(shoulder2-shoulder1)
    distanza_shoulder_pelvis = np.linalg.norm(shoulder2-pelvis2)
    offsetSX = int(rangeSX*distanza_shoulder_shoulder)
    offsetSY = int(rangeSY*distanza_shoulder_pelvis)
    
    distanza_pelvis_pelvis = np.linalg.norm(pelvis2-pelvis1)
    offsetBX = int(rangeBX*distanza_pelvis_pelvis)
    offsetBY = int(rangeBY*distanza_shoulder_pelvis)
    
    shoulder1_in[0] = shoulder1[0] - offsetSX
    shoulder1_in[1] = shoulder1[1] + offsetSY
    shoulder2_in[0] = shoulder2[0] + offsetSX
    shoulder2_in[1] = shoulder2[1] + offsetSY
    
    pelvis1_in[0] = pelvis1[0] - offsetBX 
    pelvis1_in[1] = pelvis1[1] - offsetBY 
    pelvis2_in[0] = pelvis2[0] + offsetBX 
    pelvis2_in[1] = pelvis2[1] - offsetBY
    
    poly_pts = np.array([shoulder1_in, shoulder2_in, pelvis2_in, pelvis1_in], dtype=np.int32)
    
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    cv2.imwrite('mask_torso_piccolo.png', mask)
    # Calculate the average depth within the polygon
    depth_values = depth_image[mask == 255]
    depth_values_in_centimeters = depth_values / 10

    mean_depth = np.mean(depth_values_in_centimeters) if len(depth_values_in_centimeters) > 0 else 0

    return mean_depth
    

def pelvis(pelvis2, depth_image):
    depth_value = depth_image[int(pelvis2[1]), int(pelvis2[0])]

    # Converti la profondità in centimetri
    depth_value_in_centimetri = depth_value / 10

    return depth_value_in_centimetri

def PuntoCentroshoulders(shoulder1, shoulder2, depth_image):
    centro_shoulders = np.mean([shoulder1, shoulder2], axis=0)

    depth_value = depth_image[int(centro_shoulders[1]), int(centro_shoulders[0])]

    # Converti la profondità in centimetri
    depth_value_in_centimetri = depth_value / 10

    return depth_value_in_centimetri

def PuntoCollo (shoulder1,shoulder2,nose,depth_image):
    centro_shoulders = np.mean([shoulder1, shoulder2], axis=0)
    collo=np.mean([centro_shoulders, nose], axis=0)
    
    depth_value = depth_image[int(collo[1]), int(collo[0])]
    depth_value_in_centimetri = depth_value / 10

    return depth_value_in_centimetri

def SingoloPunto(shoulder1,shoulder2,pelvis1,pelvis2,depth_image):
    centro_rettangolo = np.mean([shoulder1, shoulder2, pelvis1, pelvis2], axis=0)
    centro_triangolo = np.mean([shoulder1, shoulder2, centro_rettangolo], axis=0)

    depth_value = depth_image[int(centro_triangolo[1]), int(centro_triangolo[0])]

    # Convert depth to centimeters
    depth_value_in_centimetri = depth_value / 10
    
    mask = np.zeros_like(depth_image)
    mask[int(centro_triangolo[1]), int(centro_triangolo[0])] = 255
    cv2.imwrite('mask_punto.png', mask)

    return depth_value_in_centimetri

def SuperficiePuntiUp(shoulder1,shoulder2,pelvis1,pelvis2,depth_image):
    punto_01 = np.mean([shoulder1, pelvis1], axis=0)
    punto_02 = np.mean([shoulder2, pelvis2], axis=0)

    # Calculate the polygon from keypoints
    poly_pts = np.array([shoulder1, shoulder2, punto_02, punto_01], dtype=np.int32)
    
    # Calculate the mask of the polygon
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    
    cv2.fillPoly(mask, [poly_pts], 255)
    cv2.imwrite('mask_torso.png', mask)
    # Calculate the average depth within the polygon
    depth_values = depth_image[mask == 255]
    depth_values_in_centimeters = depth_values / 10

    mean_depth = np.mean(depth_values_in_centimeters) if len(depth_values_in_centimeters) > 0 else 0

    return mean_depth

def SuperficiePuntiDown(shoulder1,shoulder2,pelvis1,pelvis2,depth_image):
    punto_01 = np.mean([shoulder1, pelvis1], axis=0)
    punto_02 = np.mean([shoulder2, pelvis2], axis=0)

    # Calculate the polygon from keypoints
    poly_pts = np.array([punto_01, punto_02, pelvis2, pelvis1], dtype=np.int32)
    
    # Calculate the mask of the polygon
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    
    cv2.fillPoly(mask, [poly_pts], 255)
    cv2.imwrite('mask_abdomen.png', mask)
    # Calculate the average depth within the polygon
    depth_values = depth_image[mask == 255]
    depth_values_in_centimeters = depth_values / 10

    mean_depth = np.mean(depth_values_in_centimeters) if len(depth_values_in_centimeters) > 0 else 0

    return mean_depth

def SuperficiePunti(shoulder1,shoulder2,pelvis1,pelvis2,depth_image):
    # Calculate the polygon from keypoints
    poly_pts = np.array([shoulder1, shoulder2, pelvis2, pelvis1], dtype=np.int32)
    
    # Calculate the mask of the polygon
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    
    cv2.fillPoly(mask, [poly_pts], 255)
    cv2.imwrite('mask_body.png', mask)

    # Calculate the median depth within the polygon
    depth_values = depth_image[mask == 255]
    # Convert depth to centimeters (assuming depth is in millimeters)
    depth_values_in_centimeters = depth_values / 10

    # Calculate the median depth
    mean_depth = np.median(depth_values_in_centimeters) if len(depth_values_in_centimeters) > 0 else 0

    return mean_depth

logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Load the data from npy files
color_data = np.load('color_data.npy')
depth_data = np.load('depth_data.npy')

# Initialize YOLO model
model_directory = r'C:\dev\intel RealSense\FramesECG\mainGraphs\yolov8n-pose.pt'
model = YOLO(model_directory)

# Prepare a list to store depth data
depth_data_list_shoulders = []
depth_data_list_collo = []
depth_data_list_singolo_punto = []
depth_data_list_superficie_punti_up = []
depth_data_list_superficie_punti_down= []
depth_data_list_superficie_punti = []
depth_data_list_torso_piccolo = []
depth_data_list_cornice = []

# List that stores the each time between two frames
mean_time = []

tempo_inizio = time.time()

for i, (color_image, depth_image) in enumerate(zip(color_data, depth_data)):
    
    START = time.time()
    
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    results = model(color_image)
    
    high_conf_boxes = []
    processed_depth = False
    
    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints
        
        # Identify high-confidence boxes
        for box in boxes:
            if box.conf >= 0.7:
                high_conf_boxes.append(box)
                
        # If we have any high-confidence boxes, process only the first keypoint in the first box
        if high_conf_boxes:
            box = high_conf_boxes[0]
            b = box.xyxy[0].to('cpu').detach().numpy().copy()
            c=box.cls

            cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255),
                          thickness=2, lineType=cv2.LINE_4)
            cv2.putText(depth_colormap, text=model.names[int(c)], org=(int(b[0]), int(b[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255),
                        thickness=2, lineType=cv2.LINE_4)
            # Find keypoints for the high-confidence box
            for keypoint in keypoints:
                k = keypoint.xy[0].to('cpu').detach().numpy().copy()
                nose=k[0]
                eye1 = k[1]
                occhio2 = k[2]
                hear1 = k[3]
                hear2 = k[4]
                shoulder1 = k[5]
                shoulder2 = k[6]
                pelvis1 = k[11]
                pelvis2 = k[12]
            
                
                elbow1 = k[7]
                elbow2 = k[8]
                polso1 = k[9]
                polso2 = k[10]
                
                
                
                print("Algorithm 0")
                depth_data_list_collo.append(PuntoCollo(shoulder1,shoulder2,nose,depth_image))
                print("Algorithm 1")
                depth_data_list_singolo_punto.append(SingoloPunto(shoulder1,shoulder2,pelvis1,pelvis2,depth_image))
                print("Algorithm 2")
                depth_data_list_superficie_punti_down.append(SuperficiePuntiDown(shoulder1,shoulder2,pelvis1,pelvis2,depth_image))
                print("Algorithm 3")
                depth_data_list_superficie_punti_up.append(SuperficiePuntiUp(shoulder1,shoulder2,pelvis1,pelvis2,depth_image))
                print("Algorithm 4")
                depth_data_list_superficie_punti.append(SuperficiePunti(shoulder1,shoulder2,pelvis1,pelvis2,depth_image))
                print("Algorithm 5")
                depth_data_list_shoulders.append(PuntoCentroshoulders(shoulder1, shoulder2, depth_image))
                print("Algorithm 6")
                depth_data_list_torso_piccolo.append(torsoPiccolo (shoulder1, shoulder2, pelvis1, pelvis2, depth_image))
                print("Algorithm 7")
                depth_data_list_cornice.append(cornice (shoulder1, shoulder2, pelvis1, pelvis2, depth_image))
                print("------------------")
                
                cv2.circle(depth_colormap, (int(shoulder1[0]), int(shoulder1[1])), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(depth_colormap, (int(shoulder2[0]), int(shoulder2[1])), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(depth_colormap, (int(pelvis1[0]), int(pelvis1[1])), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(depth_colormap, (int(pelvis2[0]), int(pelvis2[1])), radius=5, color=(0, 0, 255), thickness=-1)
                
                processed_depth = True  # Set the flag to true
                break  # Exit the loop after processing the first high-confidence keypoint

        if processed_depth:
            break  # Exit the outer loop once a depth value is processed


    annotated_frame = results[0].plot()

    cv2.imshow("color_image", annotated_frame)
    cv2.imshow("depth_image", depth_colormap)

    mean_time.append(time.time() - START)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Convert the depth data list to a numpy array
depth_data_array0 = np.array(depth_data_list_collo)
depth_data_array1 = np.array(depth_data_list_singolo_punto)
depth_data_array2 = np.array(depth_data_list_superficie_punti_down)
depth_data_array3 = np.array(depth_data_list_superficie_punti_up)
depth_data_array4 = np.array(depth_data_list_superficie_punti)
depth_data_array5 = np.array(depth_data_list_shoulders)
depth_data_array6 = np.array(depth_data_list_torso_piccolo)
depth_data_array7 = np.array(depth_data_list_cornice)

winsound.Beep(1000,1000)
# Save the one-dimensional array to a numpy file
np.save('depth_data_yolo_collo.npy', depth_data_array0)
np.save('depth_data_yolo1.npy', depth_data_array1)
np.save('depth_data_yolo2.npy', depth_data_array2)
np.save('depth_data_yolo3.npy', depth_data_array3)
np.save('depth_data_yolo4.npy', depth_data_array4)
np.save('depth_data_yolo_shoulder.npy', depth_data_array5)
np.save('depth_data_yolo_torso_piccolo.npy', depth_data_array6)
np.save('depth_data_yolo_cornice.npy', depth_data_array7)

# Optional: Print out the depth data for verification
tot = 0
for data in mean_time:
    tot += data
mean = tot / len(mean_time)

print(f"processing time:{time.time() - tempo_inizio}")
print(f"avg time: {mean}")
print(f"1max: {max(depth_data_array1)} | min: {min(depth_data_array1)}")
print(f"2max: {max(depth_data_array2)} | min: {min(depth_data_array2)}")
print(f"3max: {max(depth_data_array3)} | min: {min(depth_data_array3)}")
print(f"4max: {max(depth_data_array4)} | min: {min(depth_data_array4)}")
print(f"Len color: {len(color_data)}")
print(f"Len depth: {len(depth_data)}")
print(f"Len depthCaptured0: {len(depth_data_array0)}")
print(f"Len depthCaptured1: {len(depth_data_array1)}")
print(f"Len depthCaptured2: {len(depth_data_array2)}")
print(f"Len depthCaptured3: {len(depth_data_array3)}")
print(f"Len depthCaptured4: {len(depth_data_array4)}")

cv2.destroyAllWindows()




# inizialmente il problema era che calcolava ogni depth per ogni box per il quale trovava i keypoint
# ora si va a costruire un array in cui al suo interno ci si mette i box con probabilità >= 0.8
# (idealmente se inquadro solo una persona il box sarà 1)
# si calcola la distanza solamente per il primo box dell'array (quindi solo 1)












