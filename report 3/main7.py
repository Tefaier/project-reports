import torchvision.models.detection

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

# already trained model
# pretrained=True is legacy version
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
# to rewrite number of output classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features, num_classes=10)

print(model)

# can be used to decrease learning rate with epochs of learning
# torch.optim.lr_scheduler.StepLR()