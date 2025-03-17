import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import trange
from torch.utils.data import Dataset, DataLoader


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            weight = weight * mask

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=None):
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            self.register_parameter('bias', None)

        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.register_parameter('mask', None)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias, self.mask)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.clone().detach()
        self.y = y.clone().detach()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, bias, mask1, mask2, mask3):
        super(Net, self).__init__()
        self.hidden1 = CustomizedLinear(n_input, n_hidden, bias=bias, mask=mask1)
        self.hidden1_activation = nn.Sigmoid()
        self.hidden2 = CustomizedLinear(n_hidden, n_hidden, bias=bias, mask=mask2)
        self.hidden2_activation = nn.Sigmoid()
        self.output = CustomizedLinear(n_hidden, n_output, bias=bias, mask=mask3)

    def forward(self, input):
        out = self.hidden1(input)
        out = self.hidden1_activation(out)
        out = self.hidden2(out)
        out = self.hidden2_activation(out)
        output = self.output(out)
        return output


def load_data(x, y, batch_size, device):
    X = torch.tensor(x, dtype=torch.float64)
    Y = torch.tensor(y, dtype=torch.float64)
    device = device
    X = X.to(device)
    Y = Y.to(device)
    ds = MyDataset(X, Y)
    # dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if len(Y) == 1:
        train_data = DataLoader(ds, batch_size=int(len(Y)), shuffle=True)
        test_data = DataLoader(ds, batch_size=int(len(Y)), shuffle=True)
    else:
        train_data, test_data = torch.utils.data.random_split(ds, [math.floor(len(Y)*0.9), math.ceil(len(Y) * 0.1)])
        train_data = DataLoader(train_data, batch_size=int(len(Y)*0.9), shuffle=True)
        test_data = DataLoader(test_data, batch_size=int(len(Y) * 0.1)+1, shuffle=True)
    return train_data, test_data


def weight_init(net, layer, activation):
    for m in net.modules():
        if isinstance(m, layer):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain(activation))
    return net


def get_masks(in_list):
    for i in range(len(in_list)):
        hidden_list = in_list[i]*3
        adjacency1 = torch.ones(in_list[i], hidden_list)
        if i == 0:
            mask1 = adjacency1
        else:
            lena, widtha = mask1.size()[0], mask1.size()[1]
            lenb, widthb = adjacency1.size()[0], adjacency1.size()[1]
            leftmaxtric = torch.vstack((mask1, torch.zeros((lenb, widtha))))
            rightmaxtric = torch.vstack((torch.zeros((lena, widthb)), adjacency1))
            mask1 = torch.hstack((leftmaxtric, rightmaxtric))

        adjacency2 = torch.ones(hidden_list, hidden_list)
        if i == 0:
            mask2 = adjacency2
        else:
            lena, widtha = mask2.size()[0], mask2.size()[1]
            lenb, widthb = adjacency2.size()[0], adjacency2.size()[1]
            leftmaxtric = torch.vstack((mask2, torch.zeros((lenb, widtha))))
            rightmaxtric = torch.vstack((torch.zeros((lena, widthb)), adjacency2))
            mask2 = torch.hstack((leftmaxtric, rightmaxtric))

        adjacency3 = torch.ones(hidden_list, 1)
        if i == 0:
            mask3 = adjacency3
        else:
            lena, widtha = mask3.size()[0], mask3.size()[1]
            lenb, widthb = adjacency3.size()[0], adjacency3.size()[1]
            leftmaxtric = torch.vstack((mask3, torch.zeros((lenb, widtha))))
            rightmaxtric = torch.vstack((torch.zeros((lena, widthb)), adjacency3))
            mask3 = torch.hstack((leftmaxtric, rightmaxtric))
    return mask1, mask2, mask3


class WeightMSELoss(nn.Module):
    def __init__(self, weight):
        super(WeightMSELoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        loss = torch.mean((output-target)**2, dim=1)
        weighted_loss = self.weight * loss
        return torch.sum(weighted_loss)


def network_test(net, test_data):
    loss_func = nn.MSELoss(reduction='mean')
    for data in test_data:
        x, y = data
        loss = loss_func(net(x), y)
        print('test_loss=%f' % loss)
    return loss


def train_network(net, lr, epoch, train_data, test_data):
    loss_func = nn.MSELoss(reduction='mean')
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    loss_flag = []
    for i in range(epoch):
        for data in train_data:
            x, y = data
            opt.zero_grad()
            loss = loss_func(net(x), y)
            if len(loss_flag) < 50:
                loss_flag.append(loss.item())
            elif len(loss_flag) == 50:
                loss_flag.pop(0)
                loss_flag.append(loss.item())
            if i == 0:
                print('epoch=%d, loss=%f' % (i + 1, loss))
            loss.backward()
            opt.step()
    while loss.item() > np.mean(loss_flag):
        for i in range(50):
            for data in train_data:
                x, y = data
                opt.zero_grad()
                loss = loss_func(net(x), y)
                if len(loss_flag) < 50:
                    loss_flag.append(loss.item())
                elif len(loss_flag) == 50:
                    loss_flag.pop(0)
                    loss_flag.append(loss.item())
                loss.backward()
                opt.step()
        epoch = epoch + 50
        if epoch > 20450:
            break
    print('epoch=%d, loss=%f' % (epoch + 1, loss))
    return loss


class single_small_net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(single_small_net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden, bias=True)
        self.hidden1_activation = nn.Sigmoid()
        self.hidden2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.hidden2_activation = nn.Sigmoid()
        self.output = nn.Linear(n_hidden, n_output, bias=True)

    def forward(self, input):
        out = self.hidden1(input)
        out = self.hidden1_activation(out)
        out = self.hidden2(out)
        out = self.hidden2_activation(out)
        output = self.output(out)
        return output


if __name__ == '__main__':
    '''torch.set_default_dtype(torch.float64)
    device = 'cuda'
    # device = 'cpu'
    x = [[1, 2, 3], [3, 4, 5]]
    y = [[1, 1], [2, 2]]
    dl = load_data(x, y, 2, device)
    mask1, mask2, mask3 = get_masks([1, 2])
    mask1 = mask1.to(device).t()
    mask2 = mask2.to(device).t()
    mask3 = mask3.to(device).t()
    net = Net(3, 9, 2, True, mask1, mask2, mask3).to(device)
    net = weight_init(net, CustomizedLinear, 'sigmoid')
    net.hidden1.weight.data = torch.mul(mask1, net.hidden1.weight.data)
    net.hidden2.weight.data = torch.mul(mask2, net.hidden2.weight.data)
    net.output.weight.data = torch.mul(mask3, net.output.weight.data)
    train_network(net, 0.001, 5000, dl)
    breakpoint()'''