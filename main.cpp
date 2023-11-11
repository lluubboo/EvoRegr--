#include <chrono>
#include "EvoAPI.hpp"

using namespace std;

int main(int, char**) {

    EvoAPI api = EvoAPI("C:/Users/lubomir.balaz/Desktop/Projekty 2023/EvoRegr++/data/TestDataSpan.csv");
    api.setBoundaryConditions(10000, 500, 3);

    auto start = std::chrono::high_resolution_clock::now();
    api.predict();
    auto stop = std::chrono::high_resolution_clock::now();

    api.show_me_result();

    std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    system("pause");
    return 0;
}
