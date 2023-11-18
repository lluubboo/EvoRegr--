#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

enum class DataArrangement {
    ColumnMajor,
    RowMajor
};

template <typename T>
class Plotter {

    std::stringstream _table;
    T* _data;
    DataArrangement _data_arrangement;
    std::vector<std::string> _column_names;
    std::string _name;
    unsigned int _table_width;
    unsigned int _column_width;
    unsigned int _size;
    unsigned int _cols;
    unsigned int _rows;
    unsigned int _precision;

    void print_content();
    void print_row(unsigned int start_index, unsigned int count, int stride);
    void print_columns_header();
    void print_table_header();
    void print_endline();
    void validate_inputs_throw_exception();

    int calculate_column_width(int table_width, int cols);
    int calculate_rows(int size, int column_count);
    bool col_width_is_sufficient(auto value, unsigned int column_width);

public:
        
    Plotter(T* data, std::string name, std::vector<std::string> column_names, unsigned int table_width, unsigned int size, DataArrangement data_arrangement);
    void print_table();
    std::string get_table();
};
