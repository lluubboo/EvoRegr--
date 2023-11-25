#include <vector>
#include "IOTools.hpp"
#include "csv.hpp"
    
/**
 * Parses a CSV file and returns the number of columns and the data as a vector.
 * 
 * @param filename The path to the CSV file.
 * @return A tuple containing the number of columns and the data as a vector.
 * @throws std::runtime_error if the file cannot be opened.
 */
template<typename T>
std::tuple<int, std::vector<T>> parse_csv(const std::string& filename) {
    // Check if file exists and can be opened
    std::ifstream file(filename);
    if (!file.good()) {
        throw std::runtime_error("File cannot be opened");
    }
    file.close();

    std::vector<T> data;
    csv::CSVFileInfo info;
    info = csv::get_file_info(filename);
    csv::CSVFormat format;
    csv::CSVReader reader(filename, format.delimiter(';').header_row(0));

    for (csv::CSVRow& row : reader) {
        for (csv::CSVField& field : row) {
            data.emplace_back(field.get<T>());
        }
    }

    return make_tuple(info.n_cols, data);
}

template std::tuple<int, std::vector<int>> parse_csv(const std::string&);
template std::tuple<int, std::vector<float>> parse_csv(const std::string&);
template std::tuple<int, std::vector<double>> parse_csv(const std::string&);