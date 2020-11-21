import torch

torch.ops.load_library("libmy_op.so")


def compute(x, y, z):
    x = torch.ops.my_ops._my_op(x, torch.eye(3))
    a = x.add(y) + torch.relu(z)
    return a.sum()


inputs = [torch.randn(2, 3), torch.randn(2, 3), torch.ones(2, 3)]
trace = torch.jit.trace(compute, inputs)
print(trace.graph)
for t in inputs:
    t.requires_grad = True
r = trace(*inputs)
print(r)
r.backward()
for t in inputs:
    print(t.grad)

