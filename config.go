package main

import (
	"fmt"
	"strings"
)

type input struct{
	values []float64
	target []float64
	position int
}

type Config struct {
	Data_File               string        `json:"data_file_location"`
	Neural_Network_File     string        `json:"neural_network_file_location"`
	Output_File             string        `json:"output_file_location"`
	Log_File                string        `json:"log_file_location"`
	Training                bool          `json:"true_if_training"`
	CM_Enabled              bool          `json:"output_confusion_matrix"`
	Test_While_Training     bool          `json:"collect_training_test_data"`
	Progress_Tracker        bool          `json:"output_progress"`
	Default_Target          bool          `json:"use_default_target"`
	Hidden_Count            []int         `json:"number_of_hidden_nodes"`
	Epoch_Update            int           `json:"epoch_update"`
	Input_Count             int           `json:"number_of_input_values"`
	Hidden_Layers           int           `json:"number_of_hidden_layers"`
	Output_Count            int           `json:"number_of_output_nodes"`
	Epoch_Count             int           `json:"number_of_epochs"`
	Targets                 [][]float64   `json:"target_values"`
	Max                     float64       `json:"value_maximum"`
	Min                     float64       `json:"value_minimum"`
	Momentum                float64       `json:"momentum"`
	Learning_Rate           float64       `json:"learning_rate"`
}

func config_error_checking () error {
	error_string := "There are some config errors that need to be fixed before runtime.\n"
	errors := 0

	if len(config.Hidden_Count) != config.Hidden_Layers || config.Hidden_Layers == 0{
		errors++
		error_string += fmt.Sprintf("\t%d. You do not have the correct number of layers or hidden node counts.\n", errors)
	}
	csv_check := strings.Split(config.Data_File, ".")
	if len(csv_check) == 0 || strings.ToLower(csv_check[len(csv_check) - 1]) != "csv" {
		errors++
		error_string += fmt.Sprintf("\t%d. The data file passed in needs to be a csv file.\n", errors)
	}
	if config.Output_Count <= 0 {
		errors++
		error_string += fmt.Sprintf("\t%d. Output count must be greater than 0.\n", errors)
	}
	if config.Input_Count <= 0 {
		errors++
		error_string += fmt.Sprintf("\t%d. Input count must be greater than 0.\n", errors)
	}
	if !config.Default_Target {
		if (config.Targets == nil) {
			errors++
			error_string += fmt.Sprintf("%d. The default target flag is set to false, but no target valuse provided.\n", errors)
		} else {
			target_check := false
			for i := 0; i < len(config.Targets); i++ {
				if len(config.Targets) != len(config.Targets[i]) {
					target_check = true
				}
				for j := 0; j < len(config.Targets[i]); j++ {
					if config.Targets[i][j] > 1 || config.Targets[i][j] < 0 {
						target_check = true
					}
				}
			}
			if target_check || config.Output_Count != len(config.Targets) {
				errors++
				error_string += fmt.Sprintf("%d. The target matrix you provided is not formatted correctly.\n", errors)
			}
		}
	}
	if config.Min >= config.Max {
		errors++
		error_string += fmt.Sprintf("\t%d. The maximum must be greater than the minimum.\n", errors)
	}
	if config.Training == true {
		if config.Momentum > 1 || config.Momentum < 0 {
			errors++
			error_string += fmt.Sprintf("\t%d. Momentum must be between 0 and 1.\n", errors)
		}
		if config.Learning_Rate > 1 || config.Learning_Rate < 0 {
			errors++
			error_string += fmt.Sprintf("\t%d. Learning rate must be between 0 and 1.\n", errors)
		}
	} else {
		if config.Neural_Network_File == "" {
			errors++
			error_string += fmt.Sprintf("\t%d. You cannot perform a Test without inputing a trained neural network.\n", errors)
		}
	}
	if errors > 0 {
		return fmt.Errorf(error_string)
	} else {
		return nil
	}
}

func new_config() *Config {
	return &Config{
		Training            : true,
		Neural_Network_File : "",
		Output_File         : "",
		Log_File            : "",
		CM_Enabled          : true,
		Test_While_Training : true,
		Progress_Tracker    : true,
		Default_Target      : true,
		Epoch_Update        : 1,
		Epoch_Count         : 50,
		Momentum            : .9,
		Learning_Rate       : .1,
	}
}
