import torch

# Locad custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='checkpoints/yolov5_512_500.pt')
# Set IoU confidence
model.iou = 0.1
# Set confidence confidence
model.conf = 0.1

def det(img, size):
    # Inference
    results = model(img, size=size)
    # result
    crops = results.crop(save=False)
    return [int(x.item()) for x in crops[0]['box']] if len(crops) > 0 else False