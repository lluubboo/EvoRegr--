#pragma once
#include <vector>
#include <iostream>
#include <tuple>

namespace IOTools {
    enum class DataArrangement {
        ColumnMajor,
        RowMajor
    };
}


template<typename T>
std::tuple<int, std::vector<T>> parse_csv(const std::string&);

template<typename T>
void export_to_csv(
    const T* data,
    int datasize, 
    int cols, 
    const std::string& filename,
    const std::vector<std::string>& headers,
    IOTools::DataArrangement data_arrangement = IOTools::DataArrangement::RowMajor,
    const std::string& delimiter = ","
);