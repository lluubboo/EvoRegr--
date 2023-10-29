#include <vector>
#include "IOTools.hpp"

#define CSV_IO_NO_THREAD

#include "csv.hpp"
    
std::tuple<int, int, std::vector<double>> parse_csv(const std::string& filename) {

    std::vector<double> vectorf;
    
    // get file info
    csv::CSVFileInfo info = csv::get_file_info(filename);

    // read csv 
    csv::CSVFormat format;
    csv::CSVReader reader(filename, format.delimiter(';').header_row(0));

    for (csv::CSVRow& row : reader) {
        for (csv::CSVField& field : row) {
            vectorf.push_back(field.get<double>());
        }
    }

    return make_tuple(info.n_rows, info.n_cols, vectorf);
}
