#include "XoshiroCpp.hpp"

namespace RandomNumbers {

    template<typename Engine>
    int rand_interval_int(int const, int const, Engine&);

    template<typename Engine, typename Number>
    Number rand_interval_decimal_number(Number const, Number const, Engine&);
}