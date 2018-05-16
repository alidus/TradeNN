import torch, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.functional as F
from torch.autograd import Variable

n_epochs = 20
input_size = 1
hidden_size = 100
batch_size = 4
n_regions = 5
total_length = 200
steps_in_region = total_length // n_regions
seq_len = steps_in_region // batch_size

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


def train_model(model, n_epochs, input, target):
    predict = []
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model.cuda().double()
    for epoch in range(n_epochs):
        for timestep in range(len(input) - 1):
            model.zero_grad()
            model.hidden = model.init_hidden()
            model.cell = model.init_hidden()
            predict = model(input)
            losses = criterion(predict, target)
            losses.backward()
            optimizer.step()
        print("EPOCH: " + str(epoch))


def give_prediction(model, prediction_length, starting_x):
    predictions = []
    predict = starting_x
    with torch.no_grad():
        for timestep in range(prediction_length - 1):
            model.hidden = model.init_hidden()
            model.cell = model.init_hidden()
            predict = model(predict)
            predictions.append(predict)
            # losses = criterion(predict, trainY)
            # losses.backward()
        return predictions


def prepare_tensors(array_of_inputs, sequence_length = seq_len):
    teX, teY = prepare_dataset(array_of_inputs)
    teX = torch.from_numpy(np.reshape(teX, (sequence_length, batch_size, 1))).cuda()
    teY = torch.from_numpy(np.reshape(teY, (sequence_length, batch_size, 1))).cuda()
    return teX, teY




json_data_array = load_json_data()
timesteps = json_data_array[1]

testX, testY = prepare_tensors(json_data_array[0][:total_length], total_length // batch_size)
fig = plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
# Построить график оригинальных цен
ax1.plot(timesteps[0:200], convert_cuda_tesor_to_1dim_ndarray(testY), color="grey", label="original rates")

model = LSTMNet(input_size, hidden_size, seq_len, batch_size)


# Основной цикл
for i in range(n_regions):
    subseq = json_data_array[0][i * steps_in_region : (i+1) * steps_in_region]
    testX, testY = prepare_tensors(subseq)
    train_model(model, n_epochs, testX, testY)
    result = give_prediction(model, steps_in_region, json_data_array[0][(i+1) * steps_in_region])


ax1.plot(timesteps[0:100], convert_cuda_tesor_to_1dim_ndarray(predict.detach()), color="blue", label="prediction")

ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=.20)
plt.show()

#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.zeros(3, 5, device=device)
#     print(x)
# else:
#     x = torch.zeros(3, 5)
#     print(x)


