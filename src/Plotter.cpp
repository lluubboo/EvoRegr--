#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "Plotter.hpp"

template <typename T>
Plotter<T>::Plotter(T* data, std::string name, std::vector<std::string> column_names, unsigned int table_width, unsigned int size, DataArrangement data_arrangement)
    : _data(data), _data_arrangement(data_arrangement), _column_names(column_names), _name(name), _table_width(table_width), _size(size) {
    validate_inputs_throw_exception();
    _cols = column_names.size();
    _rows = calculate_rows(size, _cols);
    _column_width = calculate_column_width(table_width, _cols);
}

template <typename T>
void Plotter<T>::print_table() {
    print_table_header();
    print_columns_header();
    print_content();
    std::cout << _table.str() << '\n';
}

template <typename T>
void Plotter<T>::print_table_header() {

    int left_padding = (_table_width - _name.length() - 2) / 2;
    int right_padding = _table_width - _name.length() - 2 - left_padding;

    _table << "+" << std::string(_table_width - 2, '-') << "+" << "\n"
        << "|" << std::string(left_padding, ' ') << _name << std::string(right_padding, ' ') << "|" << "\n"
        << "+" << std::string(_table_width - 2, '-') << "+" << "\n";
}

template <typename T>
void Plotter<T>::print_columns_header() {
    _table << "|";
    for (unsigned int i = 0; i < _cols; i++) {
        std::string header = _column_names[i];
        int left_padding = (_column_width - header.length()) / 2;
        int right_padding = _column_width - header.length() - left_padding;
        _table << std::string(left_padding, ' ') << header << std::string(right_padding, ' ') << "|";
    }
    _table << "\n";
    print_endline();
}

template <typename T>
void Plotter<T>::print_content() {
    if (_data_arrangement == DataArrangement::RowMajor) {
        for (unsigned int i = 0; i < _cols; i++) {
            print_row(i * _rows, _rows, 1);
        }
    }
    else {
        for (unsigned int i = 0; i < _rows; i++) {
            print_row(i, _cols, _rows);
        }
    }
    print_endline();
}

template <typename T>
void Plotter<T>::print_row(int start_index, unsigned int count, int stride) {
    _table << "|";
    for (unsigned int j = 0; j < count; j++) {
        auto value = _data[start_index + j * stride];
        _table << std::setw(_column_width) << std::setprecision(8) << std::fixed << value << "|";
    }
    _table << "\n";
}

template <typename T>
void Plotter<T>::validate_inputs_throw_exception() {
    if (_data == nullptr) {
        std::cerr << "Error: Invalid input." << std::endl;
        throw std::invalid_argument("Invalid input to plotter table - no data.");
    }

    if (_column_names.size() == 0) {
        std::cerr << "Error: Invalid column names." << std::endl;
        throw std::invalid_argument("Invalid input to plotter table - no column names.");
    }
}

template <typename T>
int Plotter<T>::calculate_column_width(int table_width, int cols) {
    return (table_width - (cols + 1)) / cols;
}

template <typename T>
int Plotter<T>::calculate_rows(int size, int column_count) {
    return size / column_count;
}

template <typename T>
void Plotter<T>::print_endline() {
    _table << "+";
    _table << std::string(_table_width - 2, '-');
    _table << "+";
    _table << "\n";
}

// Explicit instantiation
template class Plotter<int>;
template class Plotter<double>;
template class Plotter<std::string>;































