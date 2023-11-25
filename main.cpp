#include <chrono>
#include "EvoGui.hpp"

int main(int, char**) {

    EvoView* view = new EvoView(1500, 600, "EVOREGR++ 1.0.0");
    view->show();
    return Fl::run();
}
