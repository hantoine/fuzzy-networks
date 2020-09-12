#!/usr/bin/env python
# coding: utf-8

import gzip
import pickle
from absl import flags
from absl import app
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

FLAGS = flags.FLAGS
flags.DEFINE_string('device', 'cpu', 'Device used by PyTorch')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train for')
flags.DEFINE_integer('log_every', 200, 'Number of training steps between each log')
flags.DEFINE_string('model_path', 'trained_mlp_numpy.pt', 'Path where to save the trained model\'s weihghts')
flags.DEFINE_integer('batch_size', 64, 'Training batch size')


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def load_mnist(shape=(784,)):
    def make_dataset(data):
        return TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]))

    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    return (make_dataset(x) for x in (train_data, val_data, test_data))


def evaluate(net, eval_dataloader):
    metrics = {}
    n_correct = 0
    tot_samples = 0
    for data in eval_dataloader:
        inputs, labels = (x.to(FLAGS.device) for x in data)
        outputs = net(inputs)
        n_correct += (outputs.argmax(dim=1) == labels).sum().item()
        tot_samples += outputs.size(0)
    metrics['val_acc'] = n_correct / tot_samples
    return metrics


def main(_):
    net = MLP()
    train_ds, eval_ds, test_ds = load_mnist()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-2, momentum=0.0)
    criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_ds, batch_size=FLAGS.batch_size,
                                  shuffle=True, num_workers=2)
    eval_dataloader = DataLoader(eval_ds, batch_size=2*FLAGS.batch_size,
                                 shuffle=False, num_workers=4)

    net.to(FLAGS.device)
    metrics = {}
    for epoch in range(FLAGS.epochs):
        running_loss = 0.0
        n_correct = 0
        tqdm_iter = tqdm(train_dataloader)
        for i, data in enumerate(tqdm_iter):
            inputs, labels = (x.to(FLAGS.device) for x in data)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            n_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_loss += loss.item()
            if i % FLAGS.log_every == 0:
                tot_samples = FLAGS.log_every * FLAGS.batch_size
                metrics['loss'] = running_loss / tot_samples
                metrics['acc'] = n_correct / tot_samples
                running_loss = 0.0
                n_correct = 0
                tqdm_iter.set_postfix(**metrics)
        metrics.update(evaluate(net, eval_dataloader))
        tqdm_iter.set_postfix(**metrics)
    torch.save(net.state_dict(), FLAGS.model_path)


if __name__ == '__main__':
    app.run(main)
