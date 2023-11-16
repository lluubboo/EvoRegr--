#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "Plotter.hpp"

/**
 * The function `print_table` prints a table with the given data, column names, header, and data
 * arrangement.
 * 
 * @param data A pointer to an array of double values representing the data to be printed in the table.
 * @param size The `size` parameter represents the number of elements in the `data` array.
 * @param table_width The `table_width` parameter represents the total width of the table in
 * characters.
 * @param column_names A vector of strings representing the names of the columns in the table.
 * @param header The "header" parameter is a string that represents the title or header of the table.
 * It is printed at the top of the table.
 * @param data_arrangement The `data_arrangement` parameter is an enum type that specifies how the data
 * should be arranged in the table. It can have the following values:
 */
void Plotter::print_table(double* data, unsigned int size, unsigned int table_width, std::vector<std::string> column_names, std::string header, DataArrangement data_arrangement) {
    validate_inputs_throw_exception(data, column_names);
    print_table_header(header, table_width);
    print_columns_header(table_width, calculate_column_width(table_width, column_names.size()), column_names);
    print_content_based_on_arrangement(data, size, column_names.size(), table_width, data_arrangement);
    print_endline(table_width);
}

void Plotter::print_string_cell(std::string content, std::string header, unsigned int table_width) {
    print_table_header(header, table_width);
    int left_padding = (table_width - content.length() - 2) / 2;
    int right_padding = table_width - content.length() - 2 - left_padding;
    std::cout << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
    std::cout << "|" << std::string(left_padding, ' ') << content << std::string(right_padding, ' ') << "|" << std::endl;
    std::cout << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
}

/**
 * The function `print_table_header` prints a table header with a specified width and centered text.
 * 
 * @param header The "header" parameter is a string that represents the title or header of the table.
 * It is the text that will be displayed in the center of the table.
 * @param table_width The `table_width` parameter is the total width of the table, including the
 * borders.
 */
void Plotter::print_table_header(std::string header, int table_width) {

    int left_padding = (table_width - header.length() - 2) / 2;
    int right_padding = table_width - header.length() - 2 - left_padding;

    std::cout << "+" << std::string(table_width - 2, '-') << "+" << std::endl
        << "|" << std::string(left_padding, ' ') << header << std::string(right_padding, ' ') << "|" << std::endl
        << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
}

/**
 * The function `print_columns_header` prints the header row of a table with specified column names and
 * widths.
 * 
 * @param table_width The `table_width` parameter represents the total width of the table, including
 * the borders.
 * @param column_width The `column_width` parameter represents the width of each column in the table.
 * @param column_names A vector of strings containing the names of the columns in the table.
 */
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

/**
 * The function `print_content_based_on_arrangement` prints the content of a given data array based on
 * the specified data arrangement (either column major or row major).
 * 
 * @param data The `data` parameter is a pointer to an array of `double` values. It represents the data
 * that needs to be printed.
 * @param size The size parameter represents the total number of elements in the data array.
 * @param cols The parameter "cols" represents the number of columns in the data arrangement.
 * @param table_width The `table_width` parameter represents the total width of the table in
 * characters.
 * @param data_arrangement The parameter `data_arrangement` is an enum type `DataArrangement` which
 * specifies how the data should be arranged when printing. It can have two possible values:
 * `DataArrangement::ColumnMajor` and `DataArrangement::RowMajor`.
 */
void Plotter::print_content_based_on_arrangement(double* data, int size, int cols, int table_width, DataArrangement data_arrangement) {
    if (data_arrangement == DataArrangement::ColumnMajor) {
        print_content_colwise(
            data,
            cols,
            calculate_rows(size, cols),
            calculate_column_width(table_width, cols)
        );
    }
    else {
        print_content_rowwise(
            data,
            cols,
            calculate_rows(size, cols),
            calculate_column_width(table_width, cols)
        );
    }
}

/**
 * The function `print_content_colwise` prints the content of a 2D array column-wise.
 * 
 * @param data The "data" parameter is a pointer to a double array that contains the data to be
 * printed.
 * @param cols The parameter "cols" represents the number of columns in the data array.
 * @param rows The parameter "rows" represents the number of rows in the data matrix.
 * @param column_width The `column_width` parameter specifies the width of each column when printing
 * the content.
 */
void Plotter::print_content_colwise(double* data, int cols, int rows, int column_width) {
    for (int i = 0; i < rows; i++) {
        print_row(data, i, cols, rows, column_width);
    }
}

/**
 * The function `print_content_rowwise` prints the content of a 2D array row by row.
 * 
 * @param data The `data` parameter is a pointer to a double array that contains the data to be
 * printed. Each element in the array represents a value in the table.
 * @param cols The parameter "cols" represents the number of columns in the data array.
 * @param rows The parameter "rows" represents the number of rows in the data matrix.
 * @param column_width The parameter "column_width" represents the width of each column when printing
 * the content row-wise. It determines how many characters are allocated for each element in the
 * printed output.
 */
void Plotter::print_content_rowwise(double* data, int cols, int rows, int column_width) {
    for (int i = 0; i < rows; i++) {
        print_row(data, i * cols, cols, 1, column_width);
    }
}

/**
 * The function `print_row` prints a row of data values with a specified starting index, count, stride,
 * and column width.
 * 
 * @param data The `data` parameter is a pointer to an array of double values.
 * @param start_index The start_index parameter is the index of the first element in the data array
 * that should be printed.
 * @param count The parameter "count" represents the number of elements to be printed in the row.
 * @param stride The stride parameter determines the step size between consecutive elements in the data
 * array. It specifies how many elements to skip in order to get to the next element to be printed.
 * @param column_width The `column_width` parameter specifies the width of each column when printing
 * the data.
 */
void Plotter::print_row(double* data, int start_index, int count, int stride, int column_width) {
    std::cout << "|";
    for (int j = 0; j < count; j++) {
        double value = data[start_index + j * stride];
        std::cout << std::setw(column_width) << std::setprecision(8) << std::fixed << value << "|";
    }
    std::cout << std::endl;
}

/**
 * The function calculates the width of each column in a table based on the total table width and the
 * number of columns.
 * 
 * @param table_width The total width of the table, including any padding or margins.
 * @param cols The parameter "cols" represents the number of columns in the table.
 * 
 * @return the calculated column width.
 */
int Plotter::calculate_column_width(int table_width, int cols) {
    return (table_width - (cols + 1)) / cols;
}

/**
 * The function calculates the number of rows needed to display a given number of elements in a
 * specified number of columns.
 * 
 * @param size The total number of elements or items that need to be plotted.
 * @param column_count The number of columns in the plot.
 * 
 * @return the number of rows that can be created based on the given size and column count.
 */
int Plotter::calculate_rows(int size, int column_count) {
    return size / column_count;
}

/**
 * The function validates inputs for a plotter table and throws exceptions if the inputs are invalid.
 * 
 * @param data A pointer to an array of double values.
 * @param column_names A vector of strings representing the column names of the data table.
 */
void Plotter::validate_inputs_throw_exception(double* data, std::vector<std::string> column_names) {
    if (data == nullptr) {
        std::cerr << "Error: Invalid input." << std::endl;
        throw std::invalid_argument("Invalid input to plotter table - no data.");
    }

    if (column_names.size() == 0) {
        std::cerr << "Error: Invalid column names." << std::endl;
        throw std::invalid_argument("Invalid input to plotter table - no column names.");
    }
}

/**
 * The function prints a horizontal line with a specified width.
 * 
 * @param table_width The parameter `table_width` represents the width of the table that you want to
 * print.
 */
void Plotter::print_endline(int table_width) {
    std::cout << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
}
