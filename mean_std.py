from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# path of images
data_location = "D:\\Projects\\Ymnodos\\AppliedSciences\\Mel\\"

# resize images and convert to tensors
transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# load images using ImageFolder
image_data = torchvision.datasets.ImageFolder(data_location, transform=transforms)

# use DataLoader 
image_data_loader = DataLoader(image_data,

                               # batch size = whole dataset
                               batch_size=len(image_data),
                               shuffle=False,
                               num_workers=0)


# define mean and standard deviation
def mean_std(loader):
    images, lebels = next(iter(loader))

    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std


mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)
