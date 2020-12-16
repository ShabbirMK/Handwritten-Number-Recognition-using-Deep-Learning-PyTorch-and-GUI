import torch
import torchvision as tv

import matplotlib.pyplot as plt
import numpy as np
from random import randrange

import os
from shutil import copy, rmtree

CACHE_OUTER_LOCATION = './cache'
CACHE_INNER_LOCATION = './cache/image'


def view_classify(img, ps):
    '''
    Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
    plt.close()


def evalNNagainstOne(my_nn, dataset):
    '''
    This is a helper function for evaluation of a single digit. It takes in
    the loaded dataset and randomly chooses one of the digits and checks for
    its correctness.
    It takes the neural network as the first object and the dataset as the
    other one.
    '''
    # Choosing a random image from the dataset for testing
    images, labels = iter(dataset).next()
    index = randrange(0, len(images) - 1)
    img = images[index].view(1, 784)

    # Disables the following code from executing the part required for gradient
    # calculation. This is done to improve the speed of the execution.
    with torch.no_grad():
        logps = my_nn(img)

    # We obtained the logarithmic probabilities hence, we need to take antilog
    ps = torch.exp(logps)

    # We will now plot the graph of probabilities v/s class
    view_classify(img.view(1, 28, 28), ps)


def evalNNagainstDataset(my_nn, dataset):
    '''
    This is the main function for evaluation of the entire
    neural network against the loaded dataset.
    '''
    correct_count, all_count = 0, 0
    for images, labels in dataset:
        for i in range(len(labels)):
            # Flatten one single image
            img = images[i].view(1, 784)

            # Disables the following code from executing the part required for
            # gradient calculation. This is done to improve the speed of the
            # execution.
            with torch.no_grad():
                logps = my_nn(img)

            # We obtained the logarithmic probabilities
            # hence, we need to take antilog
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])

            # Obtaining the labels
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]

            # Updating the counts
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    # Printing the results
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


def testNN(my_nn, location):
    '''
    Main function responsible for testing the neural network against
    an input image. It takes the neural network as the first input and
    the location of the test image as the second
    '''
    if os.path.isfile(location):
        os.makedirs(CACHE_INNER_LOCATION)
        copy(location, CACHE_INNER_LOCATION)
        transform = tv.transforms.Compose(
            [tv.transforms.ToTensor(),
             tv.transforms.Normalize((0.5,), (0.5,)),
             tv.transforms.ToPILImage(),
             tv.transforms.Grayscale(num_output_channels=1),
             tv.transforms.ToTensor()])

        testcase = tv.datasets.ImageFolder(root=CACHE_OUTER_LOCATION, transform=transform)
        testloader = torch.utils.data.DataLoader(testcase, batch_size=1)

        image, labels = iter(testloader).next()
        image = image[0].view(1, 784)

        # Disables the following code from executing the part required for gradient
        # calculation. This is done to improve the speed of the execution.
        with torch.no_grad():
            logps = my_nn(image)

        # We obtained the logarithmic probabilities hence, we need to take antilog
        ps = torch.exp(logps)

        # We will now plot the graph of probabilities v/s class
        view_classify(image.view(1, 28, 28), ps)
        rmtree(CACHE_OUTER_LOCATION)

    else:
        print("The location is doesn't exist or is a directory!")
        exit(0)


def useNNagainstImage(my_nn, location, num_images):
    '''
    Main function responsible for testing the neural network against
    an input image containing multiple images. It takes the neural network
    as the first input and the location of the test image as the second
    '''
    result = []
    if os.path.isdir(location):
        copytree(location, "./test/testcase")
        transform = tv.transforms.Compose(
            [tv.transforms.ToTensor(),
             tv.transforms.Normalize((0.5,), (0.5,)),
             tv.transforms.ToPILImage(),
             tv.transforms.Grayscale(num_output_channels=1),
             tv.transforms.ToTensor()])

        testcase = tv.datasets.ImageFolder(root="./test", transform=transform)
        testloader = torch.utils.data.DataLoader(testcase, batch_size=num_images)

        images, labels = iter(testloader).next()
        for image in images:
            image = image.view(1, 784)

            # Disables the following code from executing the part required for gradient
            # calculation. This is done to improve the speed of the execution.
            with torch.no_grad():
                logps = my_nn(image)

            # We obtained the logarithmic probabilities hence, we need to take antilog
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])

            # Obtaining the labels
            pred_label = probab.index(max(probab))
            result.append(pred_label)

        rmtree("./test/testcase")
        return result

    else:
        print("The entered location is non-existent or not a directory")
        exit(0)


if __name__ == '__main__':
    print("This file just contains the functions for testing and evaluation")
    print("Run the main file for generation of the classifier or testing")
