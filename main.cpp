#include <chrono>
#include "EvoAPI.hpp"

using namespace std;

int main(int, char**) {

    std::cout << "\n\n" << "Loading data...";
    EvoAPI api = EvoAPI("C:/Users/lubomir.balaz/Desktop/Projekty 2023/EvoRegr++/data/TestDataSpan.csv");

    api.setBoundaryConditions(5000, 1000, 1);

    std::cout << "\n\n" << "Predicting..." << std::endl;

    auto start = chrono::steady_clock::now();
    api.predict();
    auto end = chrono::steady_clock::now();

    std::cout << "\n\n" << "Result is :";
    api.showMeBest();

    cout << "\n\n" << "Elapsed time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000. << "s" << "\n\n";
    system("pause");

    return 0;
}
