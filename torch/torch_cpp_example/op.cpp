#include <iostream>
#include <torch/script.h>
using namespace std;

torch::Tensor _my_op(torch::Tensor arg1, torch::Tensor arg2) {
  cout << "before arg1" << endl;
  cout << arg1 << endl;
  torch::Tensor mytensor =  arg1.clone();
  auto t_a = mytensor.accessor<float, 2>();
  for (int i = 0; i < t_a.size(0); i++) {
    for (int j = 0; j < t_a.size(1); j++) {
        t_a[i][j] += 2;
    }
  }
  cout << "returning tensor" << endl;
  cout << mytensor << endl;
  return mytensor;

}

static auto registry =
  torch::RegisterOperators("my_ops::_my_op", &_my_op);
