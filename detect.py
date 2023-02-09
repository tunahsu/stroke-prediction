import torch

def det(img, size, weight):
    # Locad custom YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
    # Set IoU confidence
    model.iou = 0.1
    # Set confidence confidence
    model.conf = 0.1

    # Inference
    results = model(img, size=size)
    # result
    crops = results.crop(save=False)
    return [int(x.item()) for x in crops[0]['box']] if len(crops) > 0 else False