import torch

torch.ops.load_library("libmy_op.so")


a = torch.randn(2, 3)
print(f"a={a}")
a.requires_grad = True
b = torch.ops.my_ops._my_op(a, torch.ones(2,3))
print(f"b={b}")
c = torch.randn(2,3)
print(f"c={c}")
d = b.add(c)
print(f"d={d}")
e = d.sum()
print(f"e={e}")
print(f"a.grad={a.grad}")
e.backward()
print(f"after backward a.grad={a.grad}")

