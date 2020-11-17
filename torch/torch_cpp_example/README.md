# Reference

https://pytorch.org/tutorials/advanced/cpp_frontend.html

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
```

```bash
./ptcpp
```
