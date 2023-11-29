#include <vector>

namespace DescriptiveStatistics {

    template <typename T>
    void check_arguments(std::vector<T> const& vector);

    template <typename T>
    void check_arguments(T* ptr, unsigned int size);

    template <typename T>
    std::vector<T> generate_vector(T* ptr, unsigned int size);

    template <typename T>
    T median(T*, unsigned int);

    template <typename T>
    T median(std::vector<T>);

    template <typename T>
    T mean(std::vector<T> const&);

    template <typename T>
    T geometric_mean(std::vector<T> const&);

    template <typename T>
    T standard_deviation(std::vector<T> const&);

    template <typename T>
    T standard_deviation(T*, unsigned int);

    template <typename T>
    std::vector<T> squared_residuals(std::vector<T>);
}