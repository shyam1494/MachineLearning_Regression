import math;
import csv;
import sys;
import re;

# Reading the file from standard input and storing to the final data list
def file_read():
    datum = []
    for i in range(3):
        data = sys.stdin.readlines(1);
        data1 = csv.reader(data);
        for i in data1:
            datum.append(i)

    dum = []
    for i in range(len(datum)):
        rex = re.split('; |, |\*|\t', datum[i][0]);
        dum.append(rex);

    final_data = [list(map(float, lst)) for lst in dum];
    print(final_data)
    print(len(final_data))
    idx = int(0.80 * len(final_data));
    train_set = final_data[:idx];
    test_set = final_data[idx:]
    return [train_set,test_set];


# Getting the output data for Training Inout and Training Output;
def redifined_dataset(data):
    training_input = [];
    training_output = [];
    for i in data:
        data_features = i[0:81];
        output_column = i[-1]
        training_input.append(data_features);
        training_output.append(output_column);
    return [training_input, training_output];


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    print("A",dataset)
    for row in dataset:
        for i in range(len(row)):
            print("row[i]",row[i])
            print("minmax",minmax[i][0])
            print("minmax[i][1]",minmax[i][1]);
            print("minmax[i][0]",minmax[i][0])
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Step No  : FeedForwarding Algorithm :
def forwardfeed(weight_matrix, bias, training_inputs):
    c = [];
    for i in range(len(training_inputs)):
        total = 0
        for j in range(len(weight_matrix)):
            total += (training_inputs[i][j] * weight_matrix[j]) + bias;
            sigmoid = 1 / (1 + math.exp(-total))
        c.append(sigmoid);
    print("length of feed forward", len(c));
    print(c[0:10])
    return c;

# Step : Define the total error :
def error(y_pred, y_real):
    result =[]
    for i in range(len(y_pred)):
        diff = y_real[i] - y_pred[i];
        result.append(diff **2);
    return result;

def rt_mean_square():
    error_list = error();
    no_of_obs = len(error_list);
    a = 0;
    for a in error_list:
        a += error_list[a];
    mean_sqError = a / no_of_obs;
    root_meansqError = mean_sqError  ** 0.5;
    return root_meansqError;

#Formula for backprogration
def backErrorForm1(y_pred, y_real):
    backerror1=[]
    for i in range(len(y_pred)):
        backerror1.append(y_pred[i] - y_real[i]);
    return backerror1;

# Formula for Backpropogation
def backErrorForm2(x):
    #x is prediction value
    backerror2=[]
    for i in range(len(x)):
        backerror2.append(x[i] * (1-x[i]));
    return backerror2;

def final_backprogationForm(derivative,error):
    back_finaldata=[];
    for i in range(len(error)):
        back_finaldata.append(derivative[i] * error[i]);
    return back_finaldata;

def end_backpropogation(res1,res2):
    end_finaldata = [];
    for i in range(len(res2)):
        end_finaldata.append(res1[i] * res2[i]);
    return end_finaldata;

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Step 1 - 4 : Data PreProcessing Steps :
    # Step1: Read the files
    files = file_read();
    # Step2: Spilt the files into train and test set
    training_dataset = files[0];
    test_dataset = files[1];
    # Step3 : Again get the features for Matrix Shape( Featues and Ouput)
    sample_set = redifined_dataset(training_dataset);  # Contains the Training Output and Test Output;
    training_inputs = sample_set[0]
    # print(training_inputs[0])
    training_outputs = sample_set[1];
    print("train_outputs", len(training_outputs));
    # Step4: Peform the data scaling
    # Calculate min and max for each column
    minmax = dataset_minmax(training_inputs)
    # Normalize columns
    normalize_dataset(training_inputs, minmax);


    # Step5 : Peform the Perceptron Calcualtion :
    weight_matrix = [0 for i in range(81)]
    bias = 0;
    learning_rate = 0.1
    predicted_output = forwardfeed(weight_matrix, bias, training_inputs);
    print("predicted_output",len(predicted_output));
    errorvalue = error(training_outputs,predicted_output);
    print("len of error_value",len(errorvalue));
    # Derive the BackPropogation;
    #final formula : form1 * form2 * form3 :
    #form1 = pred-ouptput
    #form2 = pred(1-predictiin)
    #form3 = sigmoid value;
    form1 = backErrorForm1(predicted_output,training_outputs);
    form2 = backErrorForm2(predicted_output)
    back_finalresult = final_backprogationForm(form1,form2);

    '''
    print("shape",len(back_finalresult));
    print("sample",back_finalresult[0:3])
    print("ke",len(training_inputs))
    print("tr",len(training_inputs[0]));
    print("tr", training_inputs[0]);'''
    back_endresult = final_backprogationForm(back_finalresult,predicted_output);
    print("endsmaple",back_endresult[0:3]);
    print("endsampleLength",len(back_endresult))
    back_endresult1 = back_endresult / len(predicted_output);
    print("endsample2",back_endresult1[0:3]);





'''
    #Running the batch epoch:
    for epoch in range(1):
        predicted_output = forwardfeed(weight_matrix, bias, training_inputs);
        rmse = rt_mean_square();
        form1 = backErrorForm1(predicted_output, training_outputs);
        form2 = backErrorForm2(predicted_output)
        back_finalresult = final_backprogationForm(form1, form2);'''
