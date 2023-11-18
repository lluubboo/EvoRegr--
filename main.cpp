#include <chrono>
#include "EvoAPI.hpp"

using namespace std;

int main(int, char**) {

    EvoAPI api = EvoAPI(100, 100, 1);
    api.load_file();
    api.predict();
    api.show_result();

    system("pause");
    return 0;
}
