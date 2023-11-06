#include <chrono>
#include "EvoAPI.hpp"

using namespace std;

int main(int, char**) {

    EvoAPI api = EvoAPI("C:/Users/lubomir.balaz/Desktop/Projekty 2023/EvoRegr++/data/TestDataSpan.csv");
    api.setBoundaryConditions(100, 100, 1);
    api.predict();
    api.show_me_result();

    system("pause");
    return 0;
}
