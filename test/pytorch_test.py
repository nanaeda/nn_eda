import numpy as np
import torch
from torch import nn
from typing import List


class Mlp(nn.Module):
  def __init__(self, widths):
    super().__init__()
    self.layers = [] 
    for i in range(len(widths) - 1):
      if 0 < len(self.layers):
        self.layers.append(nn.ReLU())
      self.layers.append(nn.Linear(widths[i], widths[i + 1]))
      nn.init.xavier_uniform_(self.layers[-1].weight)
    # self.layers.append(nn.Softmax(dim=-1))
    self.layers = nn.Sequential(*self.layers)

  def forward(self, x):
    return self.layers(x)

def gen_input(dim: int, target: int):
  res = -np.random.random(dim).astype(np.float32) * 2
  res[target] = -1
  return res

def gen_label(dim: int, target: int):
  res = np.zeros(dim, dtype=np.float32)
  res[target] = 1
  return res

def validate(net: Mlp, dim: int, n: int):
  np.random.seed(22222)

  total_prob = 0.0
  for _ in range(n):
    target = np.random.randint(dim)
    in_tensor = torch.tensor(np.array([gen_input(dim, target)]))
    total_prob += nn.Softmax(dim=-1)(net.forward(in_tensor)).detach().numpy()[0][target]
  return total_prob / n

def run(widths: List[int], num_epochs: int, num_samples: int, batch_size: int, learning_rate: float):
  np.random.seed(2345)

  net = Mlp(widths)

  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

  for epoch in range(num_epochs):
    # validation
    print("epoch=%2d, avg_prob=%0.6f" % (epoch, validate(net, widths[0], num_samples)))
    
    # train
    for _ in range(num_samples // batch_size):
      optimizer.zero_grad()
      
      inputs = []
      labels = []
      for _ in range(batch_size):
        target = np.random.randint(widths[0])
        inputs.append(gen_input(widths[0], target))
        labels.append(gen_label(widths[0], target))
      inputs = np.array(inputs)
      labels = np.array(labels)

      outputs = net(torch.tensor(inputs))      
      loss = loss_function(outputs, torch.tensor(labels))
      loss.backward()
      optimizer.step()

  return net


if __name__ == '__main__':
  run(
    widths=[10, 50, 50, 10],
    num_epochs=100,
    num_samples=int(1e5),
    batch_size=32,
    learning_rate=3e-3,
  )

