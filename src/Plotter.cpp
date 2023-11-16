#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "Plotter.hpp"

void Plotter::print_table(double* data, unsigned int size, unsigned int table_width, std::vector<std::string> column_names, std::string header, DataArrangement data_arrangement) {

    if (!validate_inputs(data, column_names)) {
        return;
    }

    int cols = column_names.size();
    int rows = calculate_rows(size, column_names.size());
    int column_width = calculate_column_width(table_width, cols);

    print_table_header(header, table_width);
    print_columns_header(table_width, column_width, column_names);

    if (data_arrangement == DataArrangement::ColumnMajor) {
        print_content_colwise(data, cols, rows, column_width);
    }
    else {
        print_content_rowwise(data, cols, rows, column_width);
    }

    std::cout << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
}

void Plotter::print_table_header(std::string header, int table_width) {
    int left_padding = (table_width - header.length() - 2) / 2;
    int right_padding = table_width - header.length() - 2 - left_padding;
    std::cout << "+" << std::string(table_width - 2, '-') << "+" << std::endl
        << "|" << std::string(left_padding, ' ') << header << std::string(right_padding, ' ') << "|" << std::endl
        << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
}

void Plotter::print_columns_header(int table_width, int column_width, std::vector<std::string> column_names) {
    int cols = column_names.size();
    std::cout << "|";
    for (int i = 0; i < cols; i++) {
        std::string header = column_names[i];
        int left_padding = (column_width - header.length()) / 2;
        int right_padding = column_width - header.length() - left_padding;
        std::cout << std::string(left_padding, ' ') << header << std::string(right_padding, ' ') << "|";
    }
    std::cout << "\n+" << std::string(table_width - 2, '-') << "+" << std::endl;
}

void Plotter::print_content_colwise(double* data, int cols, int rows, int column_width) {
    for (int i = 0; i < rows; i++) {
        print_row(data, i, cols, rows, column_width);
    }
}

void Plotter::print_content_rowwise(double* data, int cols, int rows, int column_width) {
    for (int i = 0; i < rows; i++) {
        print_row(data, i * cols, cols, 1, column_width);
    }
}

void Plotter::print_row(double* data, int start_index, int count, int stride, int column_width) {
    std::cout << "|";
    for (int j = 0; j < count; j++) {
        double value = data[start_index + j * stride];
        std::cout << std::setw(column_width) << std::setprecision(8) << std::fixed << value << "|";
    }
    std::cout << std::endl;
}

int Plotter::calculate_column_width(int table_width, int cols) {
    return (table_width - (cols + 1)) / cols;
}

int Plotter::calculate_rows(int size, int column_count) {
    return size / column_count;
}

bool Plotter::validate_inputs(double* data, std::vector<std::string> column_names) {
    if (data == nullptr) {
        std::cerr << "Error: Invalid input." << std::endl;
        return false;
    }

    if (column_names.size() == 0) {
        std::cerr << "Error: Invalid column names." << std::endl;
        return false;
    }

    return true;
}