from myutils import dataset_gen, dataset_load, dataset_view
from nn import nnBuild, nnTrain, nnLoad
from test_eval import evalNNagainstDataset, evalNNagainstOne
from test_eval import testNN
import argparse


def args_handler():
    '''
    Function handles commandline arguments in order to make use of options
    '''
    p = argparse.ArgumentParser(description='Handwritten Number Detection: Training, Evaluating and Testing Neural Network')
    p.add_argument('--action', default="test",
                   help='eg: --action=test')
    p.add_argument('--c', default="./trainednn.pt",
                   help='--c=./trainednn.pt')
    p.add_argument('--againstMNIST', default="False",
                   help='To test against MNIST dataset --againstMNIST=True')
    p.add_argument('--testimage', default="./test.png",
                   help='--testimage=./test.png')
    p.add_argument('--count', default="5", help="--count=5")

    args = p.parse_args()
    main(args)


def main(args):
    '''
    Driver function in order to control the flow of the program
    '''
    if args.action.lower() == "train":
        # Now we will train a neural netowrk and save the file
        train_the_neural_network()
    elif args.action.lower() == "test":
        # Now we can test the neural network against a sample image
        test_the_neural_network(args.c, args.againstMNIST, args.testimage)
    elif args.action.lower() == "evaluate":
        # Now we can evaluate the neural network against the MINST dataset
        eval_the_neural_network(args.c)
    elif args.action.lower() == "view":
        # Now we will get a snapshot of the dataset being used
        view_database(args.count)
    else:
        print("Cannot identify the entered argument")


def view_database(count):
    '''
    Helper function to view a snapshot of MINST dataset
    '''
    trainset, valset = dataset_gen()
    trainloader, valloader = dataset_load(trainset, valset)
    dataset_view(trainloader, int(count))


def train_the_neural_network():
    '''
    The main driver function for training the neural network
    '''
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    epochs = 5

    # Obtaining the dataset
    print("Obtaining the dataset")
    trainset, valset = dataset_gen()

    # Loading the dataset into iterative objects
    print("Get the data ready for training")
    trainloader, valloader = dataset_load(trainset, valset)

    # Building the neural network
    print("Building the neural network")
    my_nn = nnBuild(input_size, hidden_sizes, output_size)

    # Training the neural network
    print("Now, training the neural network")
    my_nn = nnTrain(my_nn, trainloader, epochs)


def eval_the_neural_network(nnlocation):
    '''
    The main driver function for evaluating the neutral network
    '''
    my_nn = nnLoad(nnlocation)

    # Obtaining the datasets
    trainset, valset = dataset_gen()
    trainloader, valloader = dataset_load(trainset, valset)

    # Evaluate the Neural Network against a dataset
    evalNNagainstDataset(my_nn, valloader)


def test_the_neural_network(nnlocation, againstMINST, testimageloc):
    '''
    The main driver function for testing the neural network
    '''
    my_nn = nnLoad(nnlocation)
    if againstMINST != "False":
        trainset, valset = dataset_gen()
        trainloader, valloader = dataset_load(trainset, valset)
        evalNNagainstOne(my_nn, valloader)

    else:
        testNN(my_nn, testimageloc)


if __name__ == '__main__':
    args_handler()
