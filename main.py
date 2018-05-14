import torch, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.functional as F
from torch.autograd import Variable




def load_data(filename):
    with open(filename) as f:
        learning_data = np.transpose(np.asarray(json.load(f)["dataset"]["data"]))

    x_data = learning_data[0]
    y_data = np.asarray(learning_data[1], float)
    y_data = normalize_array(y_data)
    return x_data, y_data



class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(LSTMNet, self).__init__()

        self.hidden_size = hidden_size
        self.inp = torch.nn.Linear(hidden_size, hidden_size)
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size = hidden_size, num_layers=1, dropout=dropout,
                                 bidirectional=False)
        self.out = torch.nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]

        x = x.view(seq_len, batch_size, -1)
        h0 = Variable(torch.zeros(seq_len, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(seq_len, batch_size, self.hidden_size))
        outputs, (ht, ct) = self.rnn(x, (h0, c0))
        out = outputs[-1]

        out = self.out(out.view(len(x), -1))
        out = self.relu(out)
        return out

def normalize_array(array):
    max_el = max(array)
    min_el = min(array)
    for i in range(len(array)):
        array[i] = (array[i] - min_el) / (max_el - min_el)
    return array

n_epochs = 10
n_iters = 50

model = LSTMNet(1, 5)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = np.zeros(n_epochs)
data = load_data("ratesLearn.json")

with torch.no_grad():
    inputs = torch.from_numpy(data[1]).view(1, -1)
    print(model(inputs))
    pass


# fig = plt.figure(figsize=(10, 5))
#     ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
#     ax1.plot(x_data, y_data, color="grey", label="original rates")
#     ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
#     plt.xticks(rotation=45)
#     plt.subplots_adjust(bottom=.20)
#     plt.show()

#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.zeros(3, 5, device=device)
#     print(x)
# else:
#     x = torch.zeros(3, 5)
#     print(x)


