// https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
// mkdir build
// cd build
// cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
#include <iostream>
#include <torch/script.h>
using namespace std;

torch::Tensor _my_op(torch::Tensor arg1, torch::Tensor arg2) {
  cout << arg1 << endl;
  cout << arg2 << endl;

  return arg1.clone();
}

static auto registry =
  torch::RegisterOperators("my_ops::_my_op", &_my_op);
