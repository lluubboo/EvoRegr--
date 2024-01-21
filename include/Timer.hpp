#include <chrono>
#include <iostream>

class Timer {

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> process_duration;
    std::chrono::duration<double> repetition_mean_duration;

    size_t repetitions;
    std::string test_name;

    void log_result() {
        std::cout << "\n";
        std::cout << "************************************TIME MEASURE*********************************";
        std::cout << "\n\n";
        std::cout << "Test Name: " << test_name;
        std::cout << "\n";
        std::cout << "Total Time Elapsed: " << process_duration.count() << " s (" << process_duration.count() * 1e6 << " us)";
        std::cout << "\n";
        std::cout << "Mean Repetition Time: " << repetition_mean_duration.count() << " s (" << repetition_mean_duration.count() * 1e6 << " us)";
        std::cout << "\n\n";
        std::cout << "*********************************************************************************";
        std::cout << "\n";
    }

public:
    Timer() : repetitions(1), test_name("Test") {
        start = std::chrono::high_resolution_clock::now();
    }

    Timer(size_t repetitions, std::string test_name) : repetitions(repetitions), test_name(test_name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {

        end = std::chrono::high_resolution_clock::now();

        process_duration = end - start;
        repetition_mean_duration = process_duration / repetitions;

        log_result();
    }
};