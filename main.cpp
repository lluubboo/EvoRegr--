#include <chrono>
#include "EvoGui.hpp"

using namespace std;

int main(int, char**) {

    EvoView* view = new EvoView(1500, 600, "EVOREGR++ 1.0.0");
    view->show();
    return Fl::run();
}
