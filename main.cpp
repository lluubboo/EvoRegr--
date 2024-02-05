#include <chrono>
#include "IEvoAPI.hpp"
#include "EvoCore.hpp"
#include "EvoGui.hpp"
#include "Log.hpp"

int main(int, char**) {
    
    // initialize logger
    EvoRegression::Log::init();

    // initialize GUI
    EvoView* view = new EvoView("EVOREGR++ 1.0.0");

    // connect GUI and backend
    std::unique_ptr<IEvoAPI> api = std::make_unique<EvoCore>();
    view->bind_to_backend(std::move(api));

    // render GUI
    view->show();

    return Fl::run();
}
