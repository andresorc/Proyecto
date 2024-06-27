from ultralytics import YOLO
import json
import numpy as np
import torch
#The function first converts the boxes to a numpy array
#Then it filters the XYWH format of the boxes and only remains the XY and the maximum value of the WH
#The function saves the box in a json file

def filter_box(xywh_property, save_path="box.json"):
    # Convertir a numpy.ndarray si es un torch.Tensor
    if isinstance(xywh_property, torch.Tensor):
        xywh_property = xywh_property.cpu().numpy()

    # Asegurar que sea un numpy.ndarray
    if isinstance(xywh_property, np.ndarray):
        if len(xywh_property.shape) == 1:
            xywh_property = np.expand_dims(xywh_property, axis=0)

        boxes_xywh = xywh_property.tolist()

        # Extraer x, y, max(w, h) y guardar en un nuevo formato
        filtered_boxes = []
        for box in boxes_xywh:
            x, y, w, h = box[:4]
            max_dim = max(w, h)
            filtered_box = {
                "x": float(x),
                "y": float(y),
                "max_dim": float(max_dim)
            }
            filtered_boxes.append(filtered_box)

        # Guardar el resultado en un archivo JSON
        with open(save_path, "w") as f:
            json.dump(filtered_boxes, f, indent=4)
        
        return filtered_boxes
    else:
        print("Error: La propiedad xywh no es un numpy.ndarray o torch.Tensor.")
        return None
    




# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["PFG/Modelos/bus.jpg"])  # return a list of Results objects


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes
    # Extract and save bounding boxes in XYWH format
    xywh = r.boxes.xywh
    filtered_boxes = filter_box(xywh, save_path="box.json")
