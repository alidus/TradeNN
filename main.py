import torch, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.functional as F
from torch.autograd import Variable


def load_json_data():
    with open("ratesLearn.json") as f:
        learning_data = np.transpose(np.asarray(json.load(f)["dataset"]["data"]))
        timesteps = np.asarray(learning_data[0])
        dataset = normalize_array(np.asarray(learning_data[1], float))

        return dataset, timesteps


def prepare_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i]
        dataX.append(a)
        dataY.append(dataset[i+1])
    a = dataset[len(dataset) - 1]
    dataX.append(a)
    dataY.append(dataset[0])
    return np.array(dataX), np.array(dataY)


class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, batch_size):
        super(LSTMNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.output = torch.nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()
        self.cell = self.init_hidden()

    def forward(self, x):
        x = x.view(self.seq_len, self.batch_size, 1)
        output, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        output = self.output(output)
        return output

    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size).double().cuda())


def normalize_array(array):
    max_el = max(array)
    min_el = min(array)
    for i in range(len(array)):
        array[i] = (array[i] - min_el) / (max_el - min_el)
    return array


def convert_cuda_tesor_to_1dim_ndarray(tensor):
    return tensor.cpu().numpy().reshape(total_length)

def train_model(model, n_epochs, input_size, hidden_size, seq_len, batch_size, trainX, trainY):
    predict = []
    for epoch in range(n_epochs):
        for timestep in range(len(trainX) - 1):
            model.zero_grad()
            model.hidden = model.init_hidden()
            model.cell = model.init_hidden()
            predict = model(trainX)
            losses = criterion(predict, trainY)
            losses.backward()
            optimizer.step()
        print("EPOCH: " + str(epoch))

def give_prediction(model, input):
    with torch.no_grad():
        prediction = model(input)
        return prediction

def ndarray_to_tensor(ndarray):
    return torch.from_numpy(np.reshape(ndarray, (seq_len, batch_size, 1))).cuda()

def one_point_prediction(model):
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)

    ax1.plot(timesteps[0:total_length], convert_cuda_tesor_to_1dim_ndarray(trainY), color="grey",
             label="original rates 1")
    result = give_prediction(model, trainX)
    ax1.plot(timesteps[0:total_length], convert_cuda_tesor_to_1dim_ndarray(result.detach()), color="blue",
             label="prediction 1")

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=.20)
    plt.show()

def seq_prediction(model, seqence_length):
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
    ax1.plot(timesteps[:len(json_data_array[0])], json_data_array[0][:len(json_data_array[0])], color="grey",
             label="original rates 1")
    first_half_pred = give_prediction(model, trainX)
    ax1.plot(timesteps[:len(json_data_array[0]) // 2], convert_cuda_tesor_to_1dim_ndarray(first_half_pred.detach()), color="blue",
             label="prediction 1")


    current_timestep = total_length
    temp_predictions = convert_cuda_tesor_to_1dim_ndarray(give_prediction(model, trainX))
    predictions = temp_predictions[:].tolist()
    while current_timestep < len(json_data_array[0]):
        for i in range(seqence_length):
            predicted_value = convert_cuda_tesor_to_1dim_ndarray(give_prediction(model, ndarray_to_tensor(temp_predictions)))[-1]
            temp_predictions = temp_predictions[1:]
            temp_predictions = np.append(temp_predictions, predicted_value)
            predictions.append(predicted_value)
            current_timestep += 1
        temp_timesteps = timesteps[current_timestep - seqence_length:current_timestep]
        ax1.plot(temp_timesteps, predictions[current_timestep - len(temp_timesteps):current_timestep], color="red",
                 label="predicted")
        temp_predictions = prepare_dataset(json_data_array[0][(current_timestep - total_length):current_timestep])[0]
        pass





    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=.20)
    plt.show()

    return predictions



n_epochs = 100
input_size = 1
hidden_size = 100
seq_len = 32
batch_size = 4
total_length = seq_len * batch_size

json_data_array = load_json_data()
timesteps = json_data_array[1]
trainX, trainY = prepare_dataset(json_data_array[0][0:total_length])

trainX = ndarray_to_tensor(trainX)

trainY = ndarray_to_tensor(trainY)

trainX = trainX.cuda()
trainY = trainY.cuda()

model = LSTMNet(input_size, hidden_size, seq_len, batch_size)
model.double()
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)



train_model(model, n_epochs, input_size, hidden_size, seq_len, batch_size, trainX, trainY)

#one_point_prediction(model)
seq_prediction(model, 20)






#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.zeros(3, 5, device=device)
#     print(x)
# else:
#     x = torch.zeros(3, 5)
#     print(x)