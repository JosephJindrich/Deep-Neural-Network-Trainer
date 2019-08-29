package main

type neural_network struct {
	Hidden_to_output_weights  [][]float64 `json:"hidden_to_output_weights"`
	Input_to_hidden_weights   [][]float64 `json:"input_to_hidden_weights"`
}

type input struct{
	values []float64
	target []float64
	position int
}


type Config struct {
	Training_Data_File      string        `json:"training_data_location"`
	Neural_Network_File     string        `json:"neural_network_file_location"`
	Output_File             string        `json:"output_file_location"`
	CM_Enabled              bool          `json:"output_confusion_matrix"`
	Test_While_Training     bool          `json:"collect_training_test_data"`
	Progress_Tracker        bool          `json:"output_progress"`
	Default_Target          bool          `json:"use_default_target"`
	Input_Count             int           `json:"number_of_input_values"`
	Hidden_Count            int           `json:"number_of_hidden_nodes"`
	Output_Count            int           `json:"number_of_output_nodes"`
	Epoch_Count             int           `json:"number_of_epochs"`
	Targets                 [][]float64   `json:"target_values"`
	Momentum                float64       `json:"momentum"`
	Learning_Rate           float64       `json:"learning_rate"`
}

func new_config() *Config {
	return &Config{
		Training_Data_File  : "train.csv",
		Neural_Network_File : "Trained_Neural_Network.txt",
		Output_File         : "Output.txt",
		CM_Enabled          : false,
		Test_While_Training : false,
		Progress_Tracker    : false,
		Default_Target      : true,
		Input_Count         : 100,
		Hidden_Count        : 10,
		Output_Count        : 2,
		Epoch_Count         : 50,
		Momentum            : .9,
		Learning_Rate       : .1,
	}
}