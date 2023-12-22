#include "EvoCore.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

EvoCore::EvoCore() {
    init_logger();
}

/**
 * Initializes the logger for the EvoAPI class.
 * If a logger with the name "EvoRegression++" already exists, it connects to it.
 * Otherwise, it creates a new logger with the name "EvoRegression++" and sets its level to debug.
 * The logger's pattern is set to "[EvoRegression++] [%H:%M:%S.%e] [%^%l%$] [thread %t] %v".
 */
void EvoCore::init_logger() {
    auto shared_logger = spdlog::get("EvoLogger");
    shared_logger->info("EvoCore logger is trying to connect to existing logger");

    if (shared_logger) {
        EvoCore::logger = shared_logger;
        EvoCore::logger->info("EvoCore logger connected to existing logger");
    }
    else {
        EvoCore::logger = spdlog::stdout_color_mt("EvoLogger");
        EvoCore::logger->set_level(spdlog::level::debug);
        EvoCore::logger->set_pattern("[EvoRegression++] [%H:%M:%S.%e] [%^%l%$] [thread %t] %v");
        EvoCore::logger->info("EvoCore logger initialized");
    }
}