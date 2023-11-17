#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "Plotter.hpp"

/**
 * @brief Constructs a Plotter object.
 * 
 * @tparam T The type of data stored in the Plotter.
 * @param data Pointer to the data array.
 * @param name The name of the Plotter.
 * @param column_names Vector of column names.
 * @param table_width The width of the table.
 * @param size The size of the data array.
 * @param data_arrangement The arrangement of the data in the table.
 */
template <typename T>
Plotter<T>::Plotter(T* data, std::string name, std::vector<std::string> column_names, unsigned int table_width, unsigned int size, DataArrangement data_arrangement)
    : _data(data), _data_arrangement(data_arrangement), _column_names(column_names), _name(name), _table_width(table_width), _size(size) {
    validate_inputs_throw_exception();
    _cols = column_names.size();
    _rows = calculate_rows(size, _cols);
    _column_width = calculate_column_width(table_width, _cols);
    _precision = 8;
}

/**
 * @brief Prints the table with headers and content.
 * 
 * This function prints the table with headers and content. It first prints the table header,
 * followed by the columns header, and then the content. Finally, it outputs the table as a string.
 * 
 * @tparam T The type of data in the table.
 */
template <typename T>
void Plotter<T>::print_table() {
    print_table_header();
    print_columns_header();
    print_content();
    std::cout << _table.str() << '\n';
}

/**
 * @brief Prints the table header.
 * 
 * This function calculates the left and right padding for the table header based on the table width and the length of the name.
 * It then prints the table header with the name centered.
 * 
 * @tparam T The type of the Plotter.
 */
template <typename T>
void Plotter<T>::print_table_header() {

    int left_padding = (_table_width - _name.length() - 2) / 2;
    int right_padding = _table_width - _name.length() - 2 - left_padding;

    _table << "+" << std::string(_table_width - 2, '-') << "+" << "\n"
        << "|" << std::string(left_padding, ' ') << _name << std::string(right_padding, ' ') << "|" << "\n"
        << "+" << std::string(_table_width - 2, '-') << "+" << "\n";
}

/**
 * @brief Prints the header of the columns in the table.
 * 
 * This function prints the header of the columns in the table. It iterates over the column names,
 * calculates the padding for each column, and prints the header with the appropriate padding.
 * 
 * @tparam T The type of data stored in the table.
 */
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

/**
 * @brief Prints the content of the Plotter object.
 * 
 * This function prints the content of the Plotter object based on the data arrangement.
 * If the data arrangement is set to RowMajor, it prints each row of the data array.
 * If the data arrangement is set to ColumnMajor, it prints each column of the data array.
 */
template <typename T>
void Plotter<T>::print_content() {
    if (_data_arrangement == DataArrangement::RowMajor) {
        for (unsigned int i = 0; i < _rows; i++) {
            print_row(i * _cols, _cols, 1);
        }
    }
    else {
        for (unsigned int i = 0; i < _rows; i++) {
            print_row(i, _cols, _rows);
        }
    }
    print_endline();
}

/**
 * @brief Prints a row of data in a table format.
 * 
 * This function prints a row of data in a table format, where each cell represents a value from the data array.
 * If a value is longer than the specified column width, it is stored in a buffer and printed at the end of the row.
 * 
 * @tparam T The type of data stored in the array.
 * @param start_index The starting index of the row in the data array.
 * @param cell_count The number of cells to print in the row.
 * @param stride The stride between consecutive cells in the data array.
 */
template <typename T>
void Plotter<T>::print_row(unsigned int start_index, unsigned int cell_count, int stride) {

    // buffer for values which are longer than the column width
    // they will be stored and printed at the end of the row
    std::stringstream too_long_values_buffer;

    _table << "|";
    for (unsigned int j = 0; j < cell_count; j++) {

        auto value = _data[start_index + j * stride];

        if (!col_width_is_sufficient(value, _column_width)) {
            too_long_values_buffer << "\n" << "cell: " << j << " value: " << value << "\n";

            // initialize to default T value
            value = T();
        }

        _table << std::setw(_column_width) << std::setprecision(_precision) << std::fixed << value << "|";
    }
    // too long values are printed without formatting because of complexity
    _table << too_long_values_buffer.str();
    _table << "\n";
}

/**
 * @brief Checks if the column width is sufficient to display the given value.
 * 
 * This function converts the value to a string with a fixed precision of _precision decimal places
 * and checks if the length of the resulting string is less than or equal to the specified column width.
 * 
 * @tparam T The type of the value.
 * @param value The value to be checked.
 * @param column_width The width of the column.
 * @return True if the column width is sufficient, false otherwise.
 */
template <typename T>
bool Plotter<T>::col_width_is_sufficient(auto value, unsigned int column_width) {
    std::stringstream ss;
    ss << std::setprecision(_precision) << std::fixed << value;
    return ss.str().length() <= column_width;
};

/**
 * @brief Validates the inputs of the Plotter class and throws an exception if they are invalid.
 * 
 * This function checks if the data pointer is null and if the column names vector is empty.
 * If either of these conditions is true, it prints an error message to the standard error stream
 * and throws an std::invalid_argument exception with an appropriate error message.
 * 
 * @tparam T The type of data stored in the Plotter.
 * @throws std::invalid_argument If the data pointer is null or the column names vector is empty.
 */
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

/**
 * @brief Calculates the column width for a table.
 * 
 * This function calculates the width of each column in a table based on the total table width and the number of columns.
 * 
 * @param table_width The total width of the table.
 * @param cols The number of columns in the table.
 * @return The calculated column width.
 */
template <typename T>
int Plotter<T>::calculate_column_width(int table_width, int cols) {
    return (table_width - (cols + 1)) / cols;
}

/**
 * @brief Calculates the number of rows needed to display a given number of elements in a specified number of columns.
 * 
 * @tparam T The type of the elements.
 * @param size The total number of elements.
 * @param column_count The number of columns.
 * @return The number of rows needed to display the elements.
 */
template <typename T>
int Plotter<T>::calculate_rows(int size, int column_count) {
    return size / column_count;
}

/**
 * @brief Prints a horizontal line with '+' at the beginning and end.
 * 
 * This method is used to print a horizontal line in the table with '+' at the beginning and end.
 * The length of the line is determined by the `_table_width` member variable.
 */
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































