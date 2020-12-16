# All the necessary imports
from torch import nn, optim, save, load
from collections import OrderedDict
from time import time
import os


def args_prep(i_size, h_sizes, o_size):
    '''
    It is a helper function in order to keep the number of hidden
    layers and number of nodes in them dynamic
    '''
    args = OrderedDict()

    # For the link between the input layer and the first hidden layer
    args['linear1'] = nn.Linear(i_size, h_sizes[0])
    args['relu1'] = nn.ReLU()

    # For the link between the hidden layers
    for index, size in enumerate(h_sizes[:-1]):
        args['linear{}'.format(str(index + 2))
             ] = nn.Linear(h_sizes[index], h_sizes[index + 1])
        args['relu{}'.format(str(index + 2))] = nn.ReLU()

    # For the link between the last hidden layer and the output layer
    args['linear{}'.format(str(index + 3))
         ] = nn.Linear(h_sizes[index + 1], o_size)
    args['relu{}'.format(str(index + 3))] = nn.ReLU()

    # Since this is a classification problem, the function to determine
    # the input belongs to which class
    args['logsoftmax'] = nn.LogSoftmax(dim=1)
    return args


def nnBuild(input_size, hidden_sizes, output_size):
    '''
    The main function for initialization of our neural network and
    setting the links. It takes in three arguments, number of input
    features as first, a list with the nodes of present in each of
    the hidden layer and the third being the number of output nodes.
    '''
    '''
    We are initializing the neural network with the bias component
    present. The first argument here is the number of nodes on
    right-hand side of the links and the second argument the number
    of nodes on the left side of the links.
    '''
    get_args = args_prep(input_size, hidden_sizes, output_size)
    model = nn.Sequential(get_args)
    return model


def nnTrain(my_nn, trainloader, epochs=25):
    '''
    The main function that will train our neural network by
    making use of SGD optimization algorithm i.e. gradient descent.
    The first argument is the initialized neural network, the
    loaded dataset is the second one and the number of iterations
    over the dataset as the third, default being 25
    '''
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(my_nn.parameters(), lr=0.003, momentum=0.9)
    start_time = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten all the MNIST images from the database into a
            # 100 * 784 vector i.e. number of images * number of pixels
            # in each image
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            # By passing the images through the neural network and
            # compute the error by checking with the correct labels
            # and the labels obtained by passing through the neural
            # network
            output = my_nn(images)
            loss = criterion(output, labels)

            # This is where the model learns using the backpropagation
            # algorithm and optimizes its weight here
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    print("\nTraining Time (in minutes) =", (time() - start_time) / 60)

    # Save the neural network so it can be furthur tested
    save(my_nn, './trainednn.pt')


def nnLoad(location):
    '''
    Simple function retrieve the neural network saved as .pt/.pth file
    '''
    if os.path.isfile(location):
        try:
            my_nn = load(location)
        except Exception:
            print("Loading of the Neural Network failed!")
            exit(0)
    else:
        print("File doesn't exist!")
        exit(0)
    return my_nn


if __name__ == '__main__':
    print("This file contains just the utility functions relating creating \
and loading of neural networks i.e. nnBuild, nnTrain and nnLoad")
    print("Run the main file for generation of the classifier or testing")
