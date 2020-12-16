import torchvision as tv
import torch
import os


CACHE_LOCATION = './cache'
NN_LOCATION = './trainednn.pt'


def useNN(nn_location, cache_location, num_rectangles):
    '''
    Main function responsible for testing the neural network against
    an input image containing multiple images. It takes the neural network
    as the first input and the location of the test image as the second
    '''
    result = []
    my_nn = torch.load(nn_location)
    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(),
         tv.transforms.Normalize((0.5,), (0.5,)),
         tv.transforms.ToPILImage(),
         tv.transforms.Grayscale(num_output_channels=1),
         tv.transforms.ToTensor()])


    testcase = tv.datasets.ImageFolder(root=cache_location, transform=transform)
    testloader = torch.utils.data.DataLoader(testcase, batch_size=num_rectangles)

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

    for root, _, files in os.walk(cache_location):
        for f in files:
            os.unlink(os.path.join(root, f))

    return result


if __name__ == '__main__':
    print(useNN(NN_LOCATION, CACHE_LOCATION, 9))
