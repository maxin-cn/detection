import torch

net = torch.nn.Conv2d(3, 256, kernel_size=3, stride=1)

gouts = []
def backward_hook(module, gin, gout):
    print(len(gin),len(gout))
    gouts.append(gout[0].data.cpu().numpy())
    gin0, gin1, gin2 = gin
    print(gin0.shape)
    print(gin1.shape)
    print(gin2.shape)


net.zero_grad()
hook = net.register_backward_hook(backward_hook)
target = torch.ones(1, 256, 126, 126)
outputs = net(torch.ones(1, 3, 128, 128))
loss = torch.sum(target - outputs)

loss.backward()
hook.remove()

