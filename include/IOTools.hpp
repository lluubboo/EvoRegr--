#pragma once
#include <vector>
#include <iostream>
#include <tuple>

template<typename T>
std::tuple<int, std::vector<T>> parse_csv(const std::string&);

std::string get_report_filename(std::string const& prefix);