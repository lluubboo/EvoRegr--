#pragma once
#include <spdlog/spdlog.h>
#include <functional>


namespace EvoRegression {

    class Log
    {
    public:
        static void init();

        static std::shared_ptr<spdlog::logger>& get_logger() { return s_logger; }

    private:
        static std::shared_ptr<spdlog::logger> s_logger;
    };

}