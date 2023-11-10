#include <chrono>
#include "EvoAPI.hpp"

using namespace std;

int main(int, char**) {

    EvoAPI api = EvoAPI("C:/Users/lubomir.balaz/Desktop/Projekty 2023/EvoRegr++/data/TestDataCube.csv");
    api.setBoundaryConditions(1000, 500, 2);
    api.predict();
    api.show_me_result();

    system("pause");
    return 0;
}
