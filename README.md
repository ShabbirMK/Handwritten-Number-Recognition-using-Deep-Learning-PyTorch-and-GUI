# Handwritten-Number-Recognition-using-Machine-Learning-and-GUI

There are two main folders:
  1. Neural Network Development - It is the main bundle of programs for training, evaluation and testing of the neural network to be used by the main application
  2. Final Application - It is the GUI based application that does the testing functionality of the above against a user specified image.

Neural Network Development: It makes use of PyTorch Framework for getting the datasets, training the neural network, and testing it against an image.

The basic commands are:
1. python project.py --action=train                    
	* Trains an NN of 128-64 nodes using backpropagation using 25 iterations
2. python project.py --action=evaluate --c="./trainednn.pt"
	* Evaluates the specified NN against the MNIST Validation Dataset
3. python project.py --action=test	--againstMNIST="True" --c="./trainednn.pt"
	* Tests the specified NN against a random image from the MNIST Dataset
4. python project.py --action=test --c="./trainednn.pt" --testimage="./1.jpg"
	* Tests the specified NN against a  user-desired image formatted according to MNIST Dataset images
5. python project.py --action=view --count=5
	* Gives a snapshot of the dataset images, the number specifies the count of the images to be displayed

Default values:
* action = test
* c = './trainednn.pt'
* againstMNIST = False
* count = 5

Final Application: It makes use of openCV to get the ROI out of the input image to segregate the image into multiple images each consisting of one image which would be individually predicted by the neural network. This process of getting the ROI can be done either using Contours or a pre-trained HAAR Classifier. The application is developed using Tkinter
