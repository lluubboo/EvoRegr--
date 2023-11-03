#include "XoshiroCpp.hpp"

namespace RandomNumbers {
    int rand_interval_int(int const, int const, XoshiroCpp::Xoshiro256Plus&);
    float rand_interval_float(float const, float const, XoshiroCpp::Xoshiro256Plus&);
}