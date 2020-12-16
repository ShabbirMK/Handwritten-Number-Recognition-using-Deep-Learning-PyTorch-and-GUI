import torch
import torchvision as tv
import matplotlib.pyplot as plt


def optimize_plot(num):
    middle = int(num ** 0.5)
    while num % middle != 0:
        middle = middle - 1

    return middle, num // middle


def dataset_view(dataset, num_of_images=6):
    '''
    This is a helper function to get a snapshot of the database
    we are working on and view characteristics of the database
    It accepts the database and the number of images present in it.
    The dataset passed must be loaded i.e. trainloader
    '''
    # To create a iterative object of the loaded dataset
    dataiter = iter(dataset)

    # To de-classify the the images and labels separately
    images, labels = dataiter.next()
    x, y = optimize_plot(num_of_images)
    image_count = 0
    fig, axes = plt.subplots(figsize=(6, 9), nrows=x, ncols=y)
    if x != 1:
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                axes[i][j].imshow(images[image_count].resize_(
                    1, 28, 28).numpy().squeeze())
                image_count += 1
                axes[i][j].axis('off')
    else:
        for i in range(len(axes)):
            axes[i].imshow(images[image_count].resize_(
                1, 28, 28).numpy().squeeze())
            image_count += 1
            axes[i].axis('off')

    plt.show()


def dataset_gen(dotransform=True):
    '''
    Utility function to generate the two datasets, the training dataset
    and the test dataset.
    It takes in one parameter, dotransform set to True by default that is
    to check if the dataset needs to be normalized or not
    '''
    if dotransform:
        '''
        torchvision.transforms is used for basic image transformations.
        transforms.ToTensor() will convert the images into array of
        numbers i.e. Tensor.
        transforms.Normalize() will normalize the images, here we are
        specifying the mean and standard deviation of the channels
        '''
        transform = tv.transforms.Compose(
            [tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,), (0.5,)), ])
    else:
        transform = None

    # Download and load the training data
    tset = tv.datasets.MNIST(root='./datasets',
                             download=True, train=True, transform=transform)
    vset = tv.datasets.MNIST(root='./datasets',
                             download=True, train=False, transform=transform)

    return tset, vset


def dataset_load(trainset, valset, train_batch_size=100, val_batch_size=100):
    '''
    Simple utility function that loads the dataset for iteration purposes
    The function accepts the datasets and converts them into objects containing
    specified number of images (default being 100)
    '''
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=val_batch_size, shuffle=True)

    return trainloader, valloader


if __name__ == '__main__':
    print("This is file only has utility functions for preparing the datasets")
    print("Run the main file for generation of the classifier or testing")
