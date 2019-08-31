package main

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
//		the hidden nodes and the weight associated with them.
// Return:	returns an array of the output nodes.
//********************************************************************

func find_outputs(network neural_network, hidden_nodes []float64) []float64 {
	var outputs []float64
	for i := 0; i < config.Output_Count; i++ {
		var dot_product float64
		dot_product = 0
		for j := 0; j < len(hidden_nodes); j++ {
			dot_product += network.Hidden_to_output_weights[i][j] * hidden_nodes[j]
		}
		outputs = append(outputs, (1 / (1 + math.Pow(2.71828, -dot_product))))
	}

	return outputs
}

//********************************************************************
// Name:	find_hidden_nodes
// Description: This function calculates the hidden nodes using the 
//		input valuse and the weights associated with them
// Return:	returns an array of the hidden nodes
//********************************************************************

func find_hidden_nodes(network neural_network, inode input) []float64 {
	var hidden_nodes []float64
	//Setting the offset
	hidden_nodes = append(hidden_nodes, 1)

	for i := 0; i < len(network.Input_to_hidden_weights); i++ {
		var dot_product float64
		dot_product = 0
		for j := 0; j < config.Input_Count; j++ {
			dot_product += inode.values[j] * network.Input_to_hidden_weights[i][j]
		}
		hidden_nodes = append(hidden_nodes, (1 / (1 + math.Pow(2.71828, -dot_product))))
	}
	return hidden_nodes
}

//********************************************************************
// Name:	create_neral_network
// Description: This function randomly assigns all the weights to x
//		where -.05 <= x <= .05 or to 0 depending on the bool
//		rand.
// Return:	returns an neural network
//********************************************************************

func create_neural_network(random bool) neural_network {
	var network neural_network
	for i := 0; i < config.Hidden_Count; i++ {
		var Input_to_hidden_weights []float64
		for j := 0; j < config.Input_Count; j++ {
			if(random){
				Input_to_hidden_weights = append(Input_to_hidden_weights, (rand.Float64() / 10) - .05)
			} else {
				Input_to_hidden_weights = append(Input_to_hidden_weights, 0)
			}
		}
		network.Input_to_hidden_weights = append(network.Input_to_hidden_weights, Input_to_hidden_weights)
	}

	for i := 0; i < config.Output_Count; i++ {
		var Hidden_to_output_weights []float64
		for j := 0; j < config.Hidden_Count + 1; j++ {
			if(random){
				Hidden_to_output_weights = append(Hidden_to_output_weights, (rand.Float64() / 10) - .05)
			} else {
				Hidden_to_output_weights = append(Hidden_to_output_weights, 0)
			}
		}
		network.Hidden_to_output_weights = append(network.Hidden_to_output_weights, Hidden_to_output_weights)
	}

	return network
}


//********************************************************************
// Name:	run_test
// Description: This function runs a test for accuracy on the given
//		data set using the current neural network. It also 
//		creates a confusion matrix when it runs.
// Return:	A string that contains the accuracy of the run and 
//		the confusion matrix.
//********************************************************************

func run_test(network neural_network, data []input) (string, [][]int) {
	hits := 0
	var confusion_matrix [][]int
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
// Return:	string - This string holds 
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
// Description: This function trains the nerual network for 50 epochs
//		and also runs a test in between every epoch for 
//		accuracy data.
// Return:	returns a trained neural network and a string for both
//		the test data and training data accuracies.
//********************************************************************

func training(training_data []input) (neural_network, string) {
	network := create_neural_network(true)
	training_str := "training data accuracy\n"

	previous_weights := create_neural_network(false)

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

			//This section prepairs the nodes for dropout to acoid overfitting
			var train_hidden_node []bool
			train_hidden_node = append(train_hidden_node, true)
			for i := 0; i < config.Hidden_Count; i++ {
				if(rand.Int() % 2 == 1) {
					train_hidden_node = append(train_hidden_node, true)
				} else {
					train_hidden_node = append(train_hidden_node, false)
				}
			}

			//here we get the error_terms for the hidden_to_output weights
			//term = output(1 - output)(target - output)
			var output_error_term []float64
			for k := 0; k < config.Output_Count; k++ {
				var dot_product float64
				dot_product = 0
				for j := 0; j < len(hidden_nodes); j++ {
					if(train_hidden_node[j]) {
						dot_product += network.Hidden_to_output_weights[k][j] * hidden_nodes[j]
					}
				}
				output := 1 / (1 + math.Pow(2.71828, -dot_product))
				output_error_term = append(output_error_term, output * (1 - output) * (training_data[data_index].target[k] - output))
			}

			//here we get the error terms for the Input_to_hidden weights
			var hidden_error_term []float64
			for j := 1; j < len(hidden_nodes); j++ {
				if(train_hidden_node[j]) {
					var dot_product float64
					dot_product = 0
					for k := 0; k < config.Output_Count; k++ {
						dot_product += network.Hidden_to_output_weights[k][j] * output_error_term[k]
					}
					hidden_error_term = append(hidden_error_term, (hidden_nodes[j] * (1 - hidden_nodes[j]) * dot_product))
				} else {
					hidden_error_term = append(hidden_error_term, 0)
				}
			}


			//adjusting the input to hidden weights using the output error term.
			for k := 0; k < config.Output_Count; k++ {
				for j := 0; j < len(hidden_nodes); j++ {
					if(train_hidden_node[j]) {
						difference := config.Learning_Rate * output_error_term[k] * hidden_nodes[j] + config.Momentum * previous_weights.Hidden_to_output_weights[k][j]
						network.Hidden_to_output_weights[k][j] += difference
						previous_weights.Hidden_to_output_weights[k][j] = difference
					}
				}
			}

			//adjusting the input to hidden weights using the hidden error term.
			for j := 0; j < config.Hidden_Count; j++ {
				if(train_hidden_node[j + 1]) {
					for i := 0; i < config.Input_Count; i++ {
						difference := config.Learning_Rate * hidden_error_term[j] * training_data[data_index].values[i] + config.Momentum * previous_weights.Input_to_hidden_weights[j][i]
						network.Input_to_hidden_weights[j][i] += difference
						previous_weights.Input_to_hidden_weights[j][i] = difference
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

	log.Print("Reading training file ", config.Training_Data_File)
	file, err := os.Open(config.Training_Data_File)
	if err != nil {
		log.Print("Error occured when opening ",
			config.Training_Data_File, "\n", err)
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
				    config.Training_Data_File + "\n\t\t", err)
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
			new_data_point.values = append(new_data_point.values,  float64(data_entry) / 255)
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
	training_data := read_csv()

	network, results := training(training_data)


	if config.Output_File != "" {
		ioutil.WriteFile(config.Output_File, []byte(results), 0644)
	} else {
		fmt.Println(results)
	}

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
	log.Print("Shutting down\n")
}
