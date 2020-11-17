#include <torch/torch.h>
#include <iostream>

int main() {
	  torch::Tensor tensor1 = torch::rand(3);
	  torch::Tensor tensor2 = torch::rand(3);
	  torch::Tensor tensor3 = tensor1.add(tensor2);
	  std::cout << tensor1 << std::endl;
	  std::cout << tensor2 << std::endl;
	  std::cout << tensor3 << std::endl;
}
