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

template <typename T>
void export_to_csv(const T* data, int datasize, int cols, const std::string& filename, const std::vector<std::string>& headers, IOTools::DataArrangement data_arrangement, const std::string& delimiter) {

    bool print_headers = true;

    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null.");
    }

    if (datasize % cols != 0) {
        throw std::invalid_argument("Data size must be a multiple of the number of columns.");
    }
    if (headers.size() != cols || headers.size() == 0) {
        print_headers = false;
        std::cerr << "Number of headers does not match the number of columns. Skipping headers." << std::endl;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file.");
    }

    if (print_headers) {
        // Write the headers
        for (int j = 0; j < cols; ++j) {
            file << headers[j];
            if (j < cols - 1) {
                file << delimiter;
            }
        }
        file << "\n";
    }

    // Write the data
    int rows = datasize / cols;
    if (data_arrangement == IOTools::DataArrangement::RowMajor) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << data[i * cols + j];
                if (j < cols - 1) {
                    file << delimiter;
                }
            }
            file << "\n";
        }
    }
    else {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << data[i + j * rows];

                if (j < cols - 1) {
                    file << delimiter;
                }
            }
            file << std::endl;
        }
    }

    file.close();
}

template void export_to_csv<double>(const double* data, int datasize, int cols, const std::string& filename, const std::vector<std::string>& headers, IOTools::DataArrangement data_arrangement, const std::string& delimiter);
template void export_to_csv<float>(const float* data, int datasize, int cols, const std::string& filename, const std::vector<std::string>& headers, IOTools::DataArrangement data_arrangement, const std::string& delimiter);
template void export_to_csv<int>(const int* data, int datasize, int cols, const std::string& filename, const std::vector<std::string>& headers, IOTools::DataArrangement data_arrangement, const std::string& delimiter);