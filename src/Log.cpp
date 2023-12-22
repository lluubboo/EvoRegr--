#include "Log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace EvoRegression {

    std::shared_ptr<spdlog::logger> Log::s_logger;

    void Log::init()
    {
        std::vector<spdlog::sink_ptr> logSinks;

        logSinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        logSinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("EvoRegression.log", true));

        s_logger = std::make_shared<spdlog::logger>("EVOREGRESSOR++", begin(logSinks), end(logSinks));
        s_logger->set_level(spdlog::level::trace);
        s_logger->flush_on(spdlog::level::trace);

        spdlog::set_pattern("[EvoRegression++] [%H:%M:%S] [%^%l%$] [thread %t] %v");
        spdlog::register_logger(s_logger);
    }
}