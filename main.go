package main
//********************************************************************
//Created by: 	Joseph Jindrich
//Last update:	09/03/19
//********************************************************************

import(
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"io"
	"io/ioutil"
	"encoding/json"
	"math"
	"os"
	"strconv"
	"math/rand"
)

var config *Config = new_config()

//********************************************************************
// Name:	find_outputs
// Description: This function calculates all the output nodes using 
//		the last layer of hidden nodes and the weights
//		associated with them.
// Return:	returns an array of the output nodes.
//********************************************************************

func find_outputs(network [][][]float64, hidden_nodes [][]float64) []float64 {
	var outputs []float64
	for i := 0; i < config.Output_Count; i++ {
		var dot_product float64
		dot_product = 0
		for j := 0; j < config.Hidden_Count[config.Hidden_Layers - 1]; j++ {
			dot_product += network[config.Hidden_Layers][i][j] * hidden_nodes[len(hidden_nodes) - 1][j]
		}
		outputs = append(outputs, (1 / (1 + math.Pow(2.71828, -dot_product))))
	}

	return outputs
}

//********************************************************************
// Name:	find_hidden_nodes
// Description: This function calculates the hidden nodes using the 
//		input valuse and the weights associated with them
// Return:	returns a 2D array of the hidden nodes
//********************************************************************

func find_hidden_nodes(network [][][]float64, inode input) [][]float64 {
	var hidden_nodes [][]float64
	// Setting the offset for each layer of hidden nodes.
	for i := 0; i < config.Hidden_Layers; i++ {
		var temp []float64
		temp = append(temp, 1)
		hidden_nodes = append(hidden_nodes, temp)
	}

	// Using the Input count to set up the first layer of Hidden nodes.
	for i := 0; i < config.Hidden_Count[0]; i++ {
		var dot_product float64
		dot_product = 0
		for j := 0; j < config.Input_Count; j++ {
			dot_product += inode.values[j] * network[0][i][j]
		}
		hidden_nodes[0] = append(hidden_nodes[0], (1 / (1 + math.Pow(2.71828, -dot_product))))
	}

	// Using each previous layer of hidden nodes to calculate the next layer of hidden nodes.
	for i := 1; i < config.Hidden_Layers; i++ {
		for j := 0; j < config.Hidden_Count[i]; j++ {
			var dot_product float64
			dot_product = 0
				for k := 0; k < config.Hidden_Count[i - 1] + 1; k++ {
					dot_product += hidden_nodes[i - 1][k] * network[i][j][k]
				}
			hidden_nodes[i] = append(hidden_nodes[i], (1 / (1 + math.Pow(2.71828, -dot_product))))
		}
	}
	return hidden_nodes
}

//********************************************************************
// Name:	create_deep_neural_network
// Description: This function randomly assigns all the weights to x
//		where -.05 <= x <= .05 or to 0 depending on the bool
//		rand.
// Return:	returns a 3D array of weights as the deep neural 
//		network
//********************************************************************

func create_deep_neural_network(random bool) [][][]float64 {
	var network [][][]float64

	// Initializing the weights from the input values, to the first hidden layer.
	var first_layer [][]float64
	for i := 0; i < config.Hidden_Count[0]; i++ {
		var new_weights []float64
		for j := 0; j < config.Input_Count; j++ {
			if(random){
				new_weights = append(new_weights, (rand.Float64() / 10) - .05)
			} else {
				new_weights = append(new_weights, 0)
			}
		}
		first_layer = append(first_layer, new_weights)
	}
	network = append(network, first_layer)

	// Initializing the weights from each previous hidden layer, to the next hidden layer.
	for i := 0; i < config.Hidden_Layers - 1; i++ {
		var new_layer [][]float64
		for j := 0; j < config.Hidden_Count[i]; j++ {
			var new_weights []float64
			for k := 0; k < config.Hidden_Count[i + 1] + 1; k++ {
				if(random){
					new_weights = append(new_weights, (rand.Float64() / 10) - .05)
				} else {
					new_weights = append(new_weights, 0)
				}
			}
			new_layer = append(new_layer, new_weights)
		}
		network = append(network, new_layer)
	}

	// Initializing the weights from the last hidden layer, to the outputs.
	var final_layer [][]float64
	for i := 0; i < config.Output_Count; i++ {
		var new_weights []float64
		for j := 0; j < config.Hidden_Count[config.Hidden_Layers - 1] + 1; j++ {
			if(random){
				new_weights = append(new_weights, (rand.Float64() / 10) - .05)
			} else {
				new_weights = append(new_weights, 0)
			}
		}
		final_layer = append(final_layer, new_weights)
	}
	network = append(network, final_layer)

	return network
}


//********************************************************************
// Name:	run_test
// Description: This function runs a test for accuracy on the given
//		data set using a deep neural network. It can also 
//		creates a confusion matrix when it runs.
// Return:	A string that contains the accuracy of the run, and 
//		the confusion matrix.
//********************************************************************

func run_test(network [][][]float64, data []input) (string, [][]int) {
	hits := 0
	var confusion_matrix [][]int
	// Initializing the confusion matrix
	if config.CM_Enabled {
		for i := 0; i < config.Output_Count; i++ {
			var new_line []int;
			for j := 0; j < config.Output_Count; j++ {
				new_line = append(new_line, 0)
			}
			confusion_matrix = append(confusion_matrix, new_line)
		}
	}

	for data_index := 0; data_index < len(data); data_index++ {
		hidden_nodes := find_hidden_nodes(network, data[data_index])
		outputs := find_outputs(network, hidden_nodes)

		// check for the highest dot product in the array
		highest_product := 0
		for input_index := 1; input_index < config.Output_Count; input_index++ {
			if outputs[highest_product] < outputs[input_index] {
				highest_product = input_index
			}
		}

		// a check to see if the neural_network was correct
		if highest_product == data[data_index].position {
			hits++
		}
		if config.CM_Enabled {
			confusion_matrix[data[data_index].position][highest_product]++
		}
	}
	return fmt.Sprintf("%4f%%", (float64(hits)/float64(len(data)) * 100)), confusion_matrix
}

//********************************************************************
// Name:	csv_styled_confusion_matrix
// Description: This function takes in a confusion matrix and converts
//		it into a csv styled string.
// Return:	A string holding the newly styled Confusion matrix.
//********************************************************************

func csv_styled_confusion_matrix(matrix [][]int) string {
	confusion_matrix := "\nConfusion Matrix\n"
	//Creating the top line of the confusion matrix.
	for i := 0; i < config.Output_Count; i++ {
		confusion_matrix += fmt.Sprintf(" ,%d", i)
	}
	confusion_matrix += fmt.Sprintf("\n")

	//generating the left column of the confusion matrix, and each rows count.
	for i := 0; i < config.Output_Count; i++ {
		confusion_matrix += fmt.Sprintf("%d, ", i)
		for j := 0; j < config.Output_Count; j++ {
			confusion_matrix += fmt.Sprintf("%d, ", matrix[i][j])
		}
		confusion_matrix += fmt.Sprintf("\n")
	}
	return confusion_matrix
}

//********************************************************************
// Name:	training
// Description: This function trains a deep nerual network for however
//		many epochs are specified in the confifg, and also 
//		runs a test in between every epoch for accuracy data.
// Return:	returns a trained neural network and a string for both
//		the accuracies.
//********************************************************************

func training(training_data []input) ([][][]float64, string) {
	network := create_deep_neural_network(true)
	training_str := "training data accuracy\n"

	previous_weights := create_deep_neural_network(false)

	for epoch_index := 0; epoch_index < config.Epoch_Count; epoch_index++ {
		if config.Test_While_Training {
			training_results, matrix := run_test(network, training_data)
			training_str += training_results
			if config.CM_Enabled {
				training_str += csv_styled_confusion_matrix(matrix)
			}
			if config.Progress_Tracker && epoch_index % config.Epoch_Update == 0 {
				log.Print("Beggining Epoch #", epoch_index, ", current accuracy is ", training_results)
			}
		} else {
			if config.Progress_Tracker && epoch_index % config.Epoch_Update == 0 {
				log.Print("Beggining Epoch #", epoch_index)
			}
		}
		for data_index := 0; data_index < len(training_data); data_index++ {

			hidden_nodes := find_hidden_nodes(network, training_data[data_index])
			// This section prepairs the nodes for dropout to avoid overfitting
			// Extra Note:
			// I'm not sure How to get this to work with a deep neural network reliably,
			// so right now it has no functionality if the hidden layer is > 1.
			var train_hidden_node [][]bool
			for i := 0; i < config.Hidden_Layers; i++ {
				var new_trainer []bool
				new_trainer = append(new_trainer, true)
				for j := 0; j < config.Hidden_Count[i]; j++ {
					if(rand.Int() % 2 == 1) {
						new_trainer = append(new_trainer, true)
					} else {
						if config.Hidden_Layers > 1 {
							new_trainer = append(new_trainer, true)
						} else {
							new_trainer = append(new_trainer, false)
						}

					}
				}
				train_hidden_node = append(train_hidden_node, new_trainer)
			}

			//here we get the error_terms for the hidden to output weights
			//term = output(1 - output)(target - output)
			var hidden_error_term [][]float64
			var output_error_term []float64
			for k := 0; k  < config.Output_Count; k++ {
				var dot_product float64
				dot_product = 0
				for j := 0; j < config.Hidden_Count[config.Hidden_Layers - 1] + 1; j++ {
					if(train_hidden_node[config.Hidden_Layers - 1][j]) {
						dot_product += network[config.Hidden_Layers][k][j] * hidden_nodes[config.Hidden_Layers - 1][j]
					}
				}
				output := 1 / (1 + math.Pow(2.71828, -dot_product))
				output_error_term = append(output_error_term, output * (1 - output) * (training_data[data_index].target[k] - output))
			}
			hidden_error_term = append(hidden_error_term, output_error_term)

			//here we get the error terms for the hidden to hidden weights
			for layer_index := config.Hidden_Layers - 1; layer_index >= 0; layer_index-- {
				var new_error_term []float64
				for j := 1; j < config.Hidden_Count[layer_index] + 1; j++ {
					if(train_hidden_node[layer_index][j]) {
						var dot_product float64
						dot_product = 0
						for k := 0; k < len(hidden_error_term[len(hidden_error_term) - 1]); k++ {
							dot_product += network[layer_index + 1][k][j] * hidden_error_term[len(hidden_error_term) - 1][k]
						}
						new_error_term = append(new_error_term, (hidden_nodes[layer_index][j] * (1 - hidden_nodes[layer_index][j]) * dot_product))
					} else {
						new_error_term = append(new_error_term, 0)
					}
				}
				hidden_error_term = append(hidden_error_term, new_error_term)
			}

			// adjusting the last hidden layers weights using the first hidden error term.
			for k := 0; k < config.Output_Count; k++ {
				var layer_index = config.Hidden_Layers - 1
				for j := 0; j < config.Hidden_Count[layer_index] + 1; j++ {
					if(train_hidden_node[layer_index][j]) {
						difference := config.Learning_Rate * hidden_error_term[0][k] * hidden_nodes[layer_index][j] +
								config.Momentum * previous_weights[config.Hidden_Layers][k][j]
						network[config.Hidden_Layers][k][j] += difference
						previous_weights[config.Hidden_Layers][k][j] = difference
					}
				}
			}

			// adjusting each hidden to hidden layer's weights using the hidden error terms.
			for layer_index := config.Hidden_Layers - 2; layer_index > 0; layer_index-- {
				for k := 0; k < config.Hidden_Count[layer_index + 1]; k++ {
					for j := 0; j < config.Hidden_Count[layer_index] + 1; j++ {
						if(train_hidden_node[layer_index][j]) {
							difference := config.Learning_Rate * hidden_error_term[(config.Hidden_Layers - 1)- layer_index][k] * hidden_nodes[layer_index][j] +
									config.Momentum * previous_weights[layer_index + 1][k][j]
							network[layer_index + 1][k][j] += difference
							previous_weights[layer_index + 1][k][j] = difference
						}
					}
				}
			}

			// adjusting the input to first hidden layer weights using the last hidden error term.
			for j := 0; j < config.Hidden_Count[0]; j++ {
				if(train_hidden_node[0][j + 1]) {
					for i := 0; i < config.Input_Count; i++ {
						difference := config.Learning_Rate * hidden_error_term[config.Hidden_Layers][j] *
							training_data[data_index].values[i] + config.Momentum * previous_weights[0][j][i]
						network[0][j][i] += difference
						previous_weights[0][j][i] = difference
					}
				}
			}
		}
	}
	if config.Progress_Tracker {
		log.Print("The final Epoch has completed")
	}
	training_str += ", \n"
	training_results, matrix := run_test(network, training_data)
	training_str += training_results
	if config.CM_Enabled {
		training_str += csv_styled_confusion_matrix(matrix)
	}
	return network, training_str
}

//********************************************************************
// Name:	read_csv
// Description: This function reads any csv file passed in, and puts
//		it's data into an array of inputs. It also takes an 
//		int that it uses to only pull a input one over that
//		int times.
// Return:	returns an array of the type input.
//********************************************************************

func read_csv() []input {
	var data []input
	var input_type_count []int
	for i:= 0; i < config.Output_Count; i++ {
		input_type_count = append(input_type_count, 0)
	}

	log.Print("Reading data file ", config.Data_File)
	file, err := os.Open(config.Data_File)
	if err != nil {
		log.Print("Error occured when opening ",
			config.Data_File, "\n", err)
		os.Exit(-1)
	}
	reader := csv.NewReader(bufio.NewReader(file))
	//a for loop that continues until it reaches the end of the file.
	for {
		line, err := reader.Read()
		//error check for the end of a file.
		if err == io.EOF {
			break
		} else if err != nil {
			log.Println("Error occured while reading through ",
				    config.Data_File + "\n\t\t", err)
			os.Exit(-1)
		}

		//Checking the Input's position in the input type array
		var new_data_point input
		new_data_point.position, err = strconv.Atoi(line[0])
		if err != nil {
			log.Print("Error occured while converting the input type on line ",
				(len(data) + 1), " of the csv input file.\n\t\t", err)
			os.Exit(-1)
		}
		if config.Default_Target {
			//setting the target values for each input type.
			for i := 0; i < config.Output_Count; i++ {
				new_data_point.target = append(new_data_point.target, .1)
			}
			new_data_point.target[new_data_point.position] = .9
		} else {
			new_data_point.target = config.Targets[new_data_point.position]
		}

		new_data_point.values = append(new_data_point.values, 1)
		//parse through each data_entry and adds it to the data point.
		for i := 1; i < len(line); i++ {
			data_entry, err := strconv.Atoi(line[i])
			if err != nil {
				log.Print("Error occured while converting input on row ", i + 1 , " on line ",
					(len(data) + 1), " of the csv input file.\n\t\t", err)
				os.Exit(-1)
			}
			new_data_point.values = append(new_data_point.values,  (float64(data_entry) - config.Min) / (config.Max - config.Min))
		}
		data = append(data, new_data_point)
		input_type_count[new_data_point.position]++
	}
	log.Print("Finished loading all training data from memory.")
	return data
}

//********************************************************************
// Name:	setup_log
// Description: This function sets up the log.
//********************************************************************

func setup_log () {
	if(config.Log_File != "") {
		log_file, err := os.OpenFile(config.Log_File, os.O_RDWR | os.O_CREATE | os.O_APPEND, 0644)
		if err != nil {
			log.Fatal("Error, Can not open ", config.Log_File , ": ", err)
		}
		log.SetOutput(log_file)
	}
}

func main() {
	var configPathFlag = flag.String("config", "./config.json", "path to configuration file")
	flag.Parse()
	if len(*configPathFlag) > 0 {
		file, err := os.Open(*configPathFlag)
		if err != nil {
			log.Fatal("Error, Can not access config: ", err)
		}

		decoder := json.NewDecoder(file)
		err = decoder.Decode(&config)
		if err != nil {
			log.Fatal("Error, Invalid config json: ", err)
		}
	}
	setup_log()
	log.Print("Starting Up")
	log.Print("Using config file ", *configPathFlag)
	err := config_error_checking()
	if err != nil {
		log.Println(err)
		os.Exit(-1)
	}

	data := read_csv()
	var network [][][]float64
	results := ""

	if config.Training {
		// if the training is set to true, it trains the neural network
		network, results = training(data)

		network_json, err := json.Marshal(network)
		if err != nil {
			log.Println("Error while marshaling The trained Nerual Network into JSON.\n", err)
			os.Exit(-1)
		}

		if config.Neural_Network_File != "" {
			ioutil.WriteFile(config.Neural_Network_File, []byte(network_json), 0644)
		} else {
			fmt.Println([]byte(network_json))
		}
	} else {
		// if the training is set to false, it tests the neural network
		log.Print("Reading Trained Neural Network File ", config.Neural_Network_File)
		file, err := ioutil.ReadFile(config.Neural_Network_File)
		if err != nil {
			log.Print("Error occured when opening ",
				config.Neural_Network_File, "\n", err)
			os.Exit(-1)
		}
		json.Unmarshal([]byte(file), &network)
		var matrix [][]int
		results, matrix = run_test(network, data)
		string_matrix := csv_styled_confusion_matrix(matrix)
		results += "\n" + string_matrix

	}

	if config.Output_File != "" {
		ioutil.WriteFile(config.Output_File, []byte(results), 0644)
	} else {
		fmt.Println(results)
	}


	log.Print("Shutting down\n")
}
