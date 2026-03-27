import torch

a = [torch.ones([30000, 30000], dtype=torch.float32).to("cuda:{}".format(i)) for i in range(8)]
b = [torch.ones([30000, 30000], dtype=torch.float32).to("cuda:{}".format(i)) for i in range(8)]
while True:
    for i in range(8):
        c = a[i] @ b[i].T