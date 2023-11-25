#include <vector>

namespace DescriptiveStatistics {

    template <typename T>
    T median(T*, int);

    template <typename T>
    T median(std::vector<T>);

    template <typename T>
    T mean(std::vector<T> const&);

    template <typename T>
    T geometric_mean(std::vector<T> const&);

    template <typename T>
    T standard_deviation(std::vector<T> const&);

    template <typename T>
    std::vector<T> squared_residuals(std::vector<T>);
}