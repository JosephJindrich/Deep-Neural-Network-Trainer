# Deep Neural Network Trainer
### Summary
This program is set up in such a way that you can run it to train a neural network using a JSON config file. There are several different variables you can pass into the file, and it will run accordingly based on those inputs. When the program finishes training the nerual network, it will turn the network into a JSON string, and output it into stdout, or a file of your choosing. This program does not apply the nerual network to a test set, it is only set up to train a network, and then output it.

This program is being converted into a fully functional Neural Network trainer, and then converted into a deep Nerual Network trainer.

### Config file inputs
**training\_data\_location** - (*string*) The file location of the training data. 
* **Notice:** The training data needs to be a .csv file.
**neural\_network\_file\_location** - (*string*) The location where the trained deep neural network will be stored when training finishes. Leaving empty prints to console.\
**output\_file\_location** - (*string*) The location where output is sent. Leaving empty prints to console.\
**collect\_training\_test\_data** - (*bool*) Set this to **true** if you want to see the Neural Networks accuracy on the training data after each epoch. The default is false.\
**output\_confusion\_matrix** - (*bool*) Set this to **true** to recieve a confusion matrix with each test data set.the default is false.\
* **Notice:** collect\_training\_test\_data must also be **true** for this to work.
**output\_progress** - (*bool*) Set this to **true** and it will log to the output location every time an epoch finished. the default is true.\
**use\_default\_target** - (*bool*) Set this to **true** to use the standard target = .9 and non\_target = .1.\
**number\_of\_input\_values** - (*int*) This is the number of values each training input has associated with it.\
**number\_of\_hidden\_nodes** - (*int*) This is the number of hidden nodes you want the neural network to have.\
**number\_of\_output\_nodes** - (*int*) This is the total number of different kinds of inputs there are.\
**number\_of\_epochs** - (*int*) The number of Epochs you want the neural netowrk to train through.\
**target\_values** - (*[][]float64*) This is the training values you want to use. The must all be > 0 and < 1, and the matrix must be a square matrix.
* **Notice:** These targets can only be used if use\_default\_targets is set to false.  
**momentum** - (*float64*) Set this to what you want the momentum to be. It must be > 0 and < 1. The default is 0.9.\
**learning\_rate** - (*float64*) Set this to what you want the learning rate to be. It must be > 0 and < 1. The default is 0.1.\




### How to format your training document
1. The format of this file needs to be a .csv. 
2. Each new input needs to be on it's own line. 
3. Each line has to have equal number of values. 
4. The first input will be an int between 0 and n(n = the varience of outputs - 1)
   * This value differentiates this input line from the others. 
   * Each input that starts with the same number should have the same target values. 
5. Every value in the input needs to be seperated with a comma and each input on a new line of the document. 

### Example data format
Example training format can be found here XXXXXXXXXXLINK\_TO\_TRAINING\_CSVXXXXXXXXXX.
