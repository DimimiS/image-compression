from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ]),
}