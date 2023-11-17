from transformers import ResNetForImageClassification

def ResNet_50(config):
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    return model
