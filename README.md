# Deep Neural Network Trainer
### Summary
This program is set up in such a way that you can run it to train a deep neural network using a JSON config file. There are several different variables you can pass into the file, and it will run accordingly based on those inputs. When the program finishes training the deep nerual network, it will turn the network into a JSON string, and output it into stdout, or a file of your choosing. This program does not apply the nerual network to a test set, it is only set up to train a network, and then output it. If you want to test your network when you finish, you can run the program with the **true_if_training** set to **false**, with all of the proper flags set, and it will test your network./

Just an asside, this Deep Neural Network works, but it is not perfect. I am not a Machine Learning expert, I did this as a fun side project. If you have any questions, or any tips let me know and I will try to answer/impliment them. 

### How to Build and Run the Software

##### Build
Requirments:\
In order to build the project, all you need is [go](https://golang.org/) installed onto your computer. 
The command to build it is\
```
go main.go config.go
```
##### Run
In order to run this software you have to create a config file. An example config file called config.json is included in the repository. The different inputs you need to have for the config file and their format are included in the Config file inputs section.\
The command to run the code is\
```
./main -conf="LOCATION_OF_CONFIG_FILE"
```
Where LOCATION\_OF\_CONF\_FILE = the location of your config file.\

### Outputs
This software has a few outputs.\
1. **Log File**: This will be some information about the program as it's running based on iputs passed in by the config file.
2. **Output File**: This is some text formatted in a csv friendly way that has information about the training as it ran.
3. **Trained Deep Neural Network**: This is the trained neural network. It will contain the neural network trained with the specifications of your config file.

### Config file inputs
**data\_file\_location** - (*string*) The file location of the dataset to be used in training or testing.
* **Notice:** The training data needs to be a .csv file.

**neural\_network\_file\_location** - (*string*) The location where the trained deep neural network will be stored when training finishes. Leaving empty prints to console.\
* **Notice:** if you have **true_if_training** set to **false** this will look for a deep neural network formated in the same way my program formats deep neural networks to use to test the data set.

**output\_file\_location** - (*string*) The location where output is sent. Leaving empty prints to console.\
**log\_file\_location** - (*string*) The location where logging is sent. Leaving empty prints to console.\
**true_if_training** - (*bool*) Setting this bool to **true** will make the program train a new neural network, and setting it to **false** will instead test a Neural Network that this program creates.\
**collect\_training\_test\_data** - (*bool*) Set this to **true** if you want to see the Neural Networks accuracy on the training data after each epoch. The default is **true**.\
**output\_confusion\_matrix** - (*bool*) Set this to **true** to recieve a confusion matrix with each test data set.the default is **false**.\
* **Notice:** collect\_training\_test\_data must also be **true** for this to work.

**output\_progress** - (*bool*) Set this to **true** and it will log to the output location every time an epoch finished. the default is **true**.\
**use\_default\_target** - (*bool*) Set this to **true** to use the standard target = .9 and non\_target = .1. The default is **true**\
**number\_of\_hidden\_nodes** - (*int*) This is an array that will hold the number of hidden nodes you want each hidden layer of the deep neural network to have.\
**number\_of\_input\_values** - (*int*) This is the number of values each training input has associated with it.\
**number\_of\_output\_nodes** - (*int*) This is the total number of different kinds of inputs there are.\
**number\_of\_hidden\_layers** - (*int*) This is the number of hidden layers you want the deep neural network to have.\
**number\_of\_epochs** - (*int*) The number of epochs you want the neural netowrk to train through. The default is 50.\
**epoch\_update** - (*int*) The number of epochs that need to complete for the log to output an update. The default is 1.
* **Notice:** output\_progress must be **true** for this to work.

**target\_values** - (*[][]float64*) This is the training values you want to use. The must all be > 0 and < 1, and the matrix must be a square matrix.
* **Notice:** These targets can only be used if use\_default\_targets is set to false.  

**minimum_value** - (*float64*) Set this to the lowest possible value of the data.\
**maximum_value** - (*float64*) Set this to the highest possible value of the data.\
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
Example training format can be found [here](https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv)
