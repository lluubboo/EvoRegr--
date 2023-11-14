#include <vector>
#include "IOTools.hpp"
#define CSV_IO_NO_THREAD
#include "csv.hpp"
    
/**
 * The function `parse_csv` reads a CSV file, extracts the data, and returns the number of rows, number
 * of columns, and the data as a vector. Errors are handled above this function.
 * 
 * @param filename The filename parameter is a string that represents the name of the CSV file that you
 * want to parse.
 * 
 * @return The function `parse_csv` returns a `std::tuple<int, int, std::vector<double>>`.
 */
std::tuple<int, int, std::vector<double>> parse_csv(const std::string& filename) {

    std::vector<double> data;
    csv::CSVFileInfo info;
    info = csv::get_file_info(filename);
    csv::CSVFormat format;
    csv::CSVReader reader(filename, format.delimiter(';').header_row(0));

    for (csv::CSVRow& row : reader) {
        for (csv::CSVField& field : row) {
            data.push_back(field.get<double>());
        }
    }

    return make_tuple(info.n_rows, info.n_cols, data);
}
