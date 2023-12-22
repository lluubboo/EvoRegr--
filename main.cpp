#include <chrono>
#include "EvoGui.hpp"
#include "Log.hpp"

int main(int, char**) {
    
    // initialize logger
    EvoRegression::Log::init();

    // initialize gui
    EvoView* view = new EvoView(1500, 600, "EVOREGR++ 1.0.0");
    view->show();
    return Fl::run();
}
