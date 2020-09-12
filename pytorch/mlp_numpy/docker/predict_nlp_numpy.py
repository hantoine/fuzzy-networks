#!/usr/bin/env python
# coding: utf-8

import gzip
import pickle
from absl import flags
from absl import app
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

flags.DEFINE_string('device', 'cpu', 'Device used by PyTorch')
flags.DEFINE_integer('batch_size', 64, 'Training batch size')
flags.DEFINE_string('predictions_path', 'predictions.pt',
                    'Path where to save the predictions on test set')
flags.DEFINE_integer('n_samples', 640, 'Run predictions on the first n samples only')
flags.DEFINE_string('model_path', 'trained_mlp_numpy.pt', 'Path where the trained model\'s weihghts are sved')
FLAGS = flags.FLAGS


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


def load_mnist(set_type='train', n_samples=None):
    def make_dataset(d):
        return TensorDataset(torch.from_numpy(d[0]), torch.from_numpy(d[1]))

    with gzip.open("mnist.pkl.gz", "rb") as data_file:
        data = pickle.load(data_file, encoding="latin1")

    data = {k: v for k, v in zip(('train', 'val', 'test'), data)}
    if n_samples is None:
        n_samples = len(data[set_type])
    return make_dataset(data[set_type][:n_samples])


def predict(net, test_dataloader):
    predictions = []
    net.to(FLAGS.device)
    for data in tqdm(test_dataloader):
        inputs, labels = (x.to(FLAGS.device) for x in data)
        outputs = net(inputs)
        predictions.append(outputs)
    return torch.cat(predictions, dim=0)


def main(_):
    net = MLP()
    net.load_state_dict(torch.load(FLAGS.model_path, map_location=FLAGS.device))
    test_ds = load_mnist(set_type='test', n_samples=FLAGS.n_samples)
    test_dataloader = DataLoader(test_ds, batch_size=FLAGS.batch_size,
                                 shuffle=False)
    predictions = predict(net, test_dataloader)
    torch.save(predictions, FLAGS.predictions_path)


if __name__ == '__main__':
    app.run(main)
