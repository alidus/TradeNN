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
    return tensor.cpu().numpy().reshape(100)

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

def give_prediction(model, num_of_regions):
    with torch.no_grad():
        for timestep in range(len(trainX) - 1):
            model.hidden = model.init_hidden()
            model.cell = model.init_hidden()
            predict = model(trainX)
            losses = criterion(predict, trainY)
            losses.backward()
            optimizer.step()
        print("EPOCH: " + str(epoch))


n_epochs = 500
input_size = 1
hidden_size = 100
seq_len = 50
batch_size = 2

json_data_array = load_json_data()
timesteps = json_data_array[1]
trainX, trainY = prepare_dataset(json_data_array[0][0:100])
testX, testY = prepare_dataset(json_data_array[0][100:200])

trainX = torch.from_numpy(np.reshape(trainX, (seq_len, batch_size, 1)))
testX = torch.from_numpy(np.reshape(testX, (seq_len, batch_size, 1)))

trainY = torch.from_numpy(np.reshape(trainY, (seq_len, batch_size, 1)))
testY = torch.from_numpy(np.reshape(testY, (seq_len, batch_size, 1)))

trainX = trainX.cuda()
trainY = trainY.cuda()
testX = trainX.cuda()
testY = trainY.cuda()

model = LSTMNet(input_size, hidden_size, seq_len, batch_size)
model.double()
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

fig = plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
ax1.plot(timesteps[0:100], convert_cuda_tesor_to_1dim_ndarray(trainY), color="grey", label="original rates")

train_model(model, n_epochs, input_size, hidden_size, seq_len, batch_size, trainX, trainY)
result = give_prediction(model)


ax1.plot(timesteps[0:100], convert_cuda_tesor_to_1dim_ndarray(predict.detach()), color="blue", label="prediction")
trainX = convert_cuda_tesor_to_1dim_ndarray(trainX)
trainY = convert_cuda_tesor_to_1dim_ndarray(trainY)

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


