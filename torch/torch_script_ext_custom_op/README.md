# Reference

https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
```

```python
torch.ops.my_ops._my_op(torch.randn(4,4), torch.randn(3,3))
torch.ops.load_library("libmy_op.so")
```
