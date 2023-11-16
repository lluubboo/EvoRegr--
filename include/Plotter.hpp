#include <iostream>
#include <iomanip>
#include <vector>

namespace Plotter {

    enum DataArrangement {
        ColumnMajor,
        RowMajor
    };

    void print_table(double* data, unsigned int size, unsigned int table_width, std::vector<std::string> column_names, std::string header, DataArrangement data_arrangement);
    void print_table_header(std::string header, int table_width);
    void print_columns_header(int table_width, int column_width, std::vector<std::string> column_names);
    void print_content_based_on_arrangement(double* data, int size, int cols, int table_width, DataArrangement data_arrangement);
    void print_content_colwise(double* data, int cols, int rows, int column_width);
    void print_content_rowwise(double* data, int cols, int rows, int column_width);
    void print_row(double* data, int start_index, int count, int stride, int column_width);
    int calculate_column_width(int table_width, int cols);
    int calculate_rows(int size, int column_count);
    void validate_inputs_throw_exception(double* data, std::vector<std::string> column_names);
    void print_endline(int table_width);
    void print_string_cell(std::string content, std::string name, unsigned int table_width);
}
