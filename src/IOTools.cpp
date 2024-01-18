#include "IOTools.hpp"
#include "csv.hpp"
    

/**
 * Parses a CSV file and returns the number of columns and the data as a vector.
 * 
 * @param filename The path to the CSV file.
 * @return A tuple containing the number of columns and the data vector.
 *         The data vector is stored in column-major order.
 * @tparam T The type of data to be parsed from the CSV file.
 */
template<typename T>
std::tuple<int, std::vector<T>> parse_csv(const std::string& filename) {

    csv::CSVFileInfo info = csv::get_file_info(filename);
    csv::CSVFormat format;
    csv::CSVReader reader(filename, format.delimiter(info.delim).header_row(0));

    info.n_rows++; // add one row for the header, probably bug in the library

    std::vector<T> data(info.n_cols * info.n_rows, 0);

    // read the data from the csv file into the data vector column major order
    size_t index_fictive = 0;
    size_t row_fictive = 0;
    for (csv::CSVRow& row : reader) {
        for (csv::CSVField& field : row) {
            data[info.n_rows * index_fictive++ + row_fictive] = field.get<T>();
        }
        row_fictive++;
        index_fictive = 0;
    }

    return make_tuple(info.n_cols, data);
}

template std::tuple<int, std::vector<int>> parse_csv(const std::string&);
template std::tuple<int, std::vector<float>> parse_csv(const std::string&);
template std::tuple<int, std::vector<double>> parse_csv(const std::string&);

/**
 * @brief Returns the filename for a regression report.
 * 
 * This function generates a filename for a regression report based on the given prefix and the current date and time.
 * The filename format is "regression_report_prefix_year-month-day_hour-minute.txt".
 * 
 * @param prefix The prefix to be included in the filename.
 * @return The generated filename for the regression report.
 */
std::string get_report_filename(std::string const& prefix) {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::string filename = prefix + "_";
    filename += std::to_string(tm.tm_year + 1900) + "-";
    filename += std::to_string(tm.tm_mon + 1) + "-";
    filename += std::to_string(tm.tm_mday) + "_";
    filename += std::to_string(tm.tm_hour) + "-";
    filename += std::to_string(tm.tm_min) + "-";
    filename += std::to_string(tm.tm_sec) + ".txt";
    return filename;
};