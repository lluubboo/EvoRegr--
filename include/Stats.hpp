#include <vector>

namespace DescriptiveStatistics {
    double median(double*, int);
    double median(std::vector<double>);
    double mean(std::vector<double> const&);
    double geometric_mean(std::vector<double> const&);
    double standard_deviation(std::vector<double> const&);
    std::vector<double> squared_residuals(std::vector<double>);
}