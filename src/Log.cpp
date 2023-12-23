#include "Log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace EvoRegression {

    std::shared_ptr<spdlog::logger> Log::s_logger;

    /**
     * Initializes the logging system.
     * This function sets up the necessary sinks for logging, including a console sink and a file sink.
     * It also configures the logger with a specific pattern and log level.
     * After calling this function, the logging system is ready to be used.
     */
    void Log::init()
    {
        std::vector<spdlog::sink_ptr> logSinks;

        logSinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        logSinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("EvoRegression.log", true));

        s_logger = std::make_shared<spdlog::logger>("EVOREGRESSOR++", begin(logSinks), end(logSinks));

        spdlog::set_pattern("[EvoRegression++] [%H:%M:%S] [%^%l%$] [thread %t] %v");
        spdlog::register_logger(s_logger);
        spdlog::set_level(spdlog::level::trace);
        spdlog::flush_on(spdlog::level::trace);
    }

    /**
     * @brief Get the logger instance.
     * 
     * @return std::shared_ptr<spdlog::logger>& The logger instance.
     */
    std::shared_ptr<spdlog::logger>& Log::get_logger() {
        if (!s_logger) {
            throw std::runtime_error("Logger not initialized");
        }
        return s_logger;
    }
}