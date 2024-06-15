#include <mlpack.hpp>
#include <armadillo>
#include "../models/models/alexnet/alexnet.hpp"
#include <iostream>

int main(void) {
	mlpack::models::AlexNet net = mlpack::models::AlexNet();	
	std::cout << "Hello world\n";
	return 0;
}
