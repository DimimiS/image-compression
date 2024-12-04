from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        # transforms.CenterCrop(512),
        transforms.ToTensor()
    ]),
}