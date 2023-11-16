#include <chrono>
#include "EvoAPI.hpp"

using namespace std;

int main(int, char**) {

    EvoAPI api = EvoAPI(100, 100, 1);
    api.load_file();

    auto start = std::chrono::high_resolution_clock::now();
    api.predict();
    auto stop = std::chrono::high_resolution_clock::now();

    api.show_result();

    std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    system("pause");
    return 0;
}
