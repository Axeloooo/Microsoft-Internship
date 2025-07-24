/**
 * @file csv_writer.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Lightweight CSV writer.
 * @version 0.1
 * @date 2025-07-04
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "utils/csv_writer.h"

#include <stdexcept>

namespace trt_allreduce::utils {

/**
 * @brief Construct a new Csv Writer object
 *
 * @param path Path to the CSV file.
 * @param with_header Whether to write the header row.
 */
CsvWriter::CsvWriter(const std::string& path, bool with_header) : file_(path, std::ios::out) {
    if (!file_) {
        throw std::runtime_error("CsvWriter: cannot open " + path);
    }
    if (with_header) {
        file_ << "implementation,data_type,iterations,message_bytes,latency_ms,bandwidth_GBps,accuracy_pct,max_abs_error,mean_abs_error,max_rel_error,rmse" << std::endl;
    }
}

/**
 * @brief Destroy the Csv Writer object
 *
 */
CsvWriter::~CsvWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

/**
 * @brief Write a row to the CSV file.
 *
 * @param impl Implementation name.
 * @param data_type Data type of the message.
 * @param iters Number of iterations.
 * @param bytes Number of bytes in the message.
 * @param latency_ms Latency in milliseconds.
 * @param bandwidth_tb Bandwidth in TBps.
 * @param accuracy_pct Accuracy percentage.
 * @param max_abs_err Maximum absolute error.
 * @param mean_abs_err Mean absolute error.
 * @param max_rel_err Maximum relative error.
 * @param rmse Root mean square error.
 */
void CsvWriter::WriteRow(const std::string& impl, const std::string& data_type, const std::size_t& iters, std::size_t bytes, double latency_ms, double bandwidth_tb,
                         double accuracy_pct, float max_abs_err, double mean_abs_err, float max_rel_err, double rmse) {
    file_ << impl << ',' << data_type << ',' << iters << ',' << bytes << ',' << latency_ms << ',' << bandwidth_tb << ',' << accuracy_pct << ',' << max_abs_err << ','
          << mean_abs_err << ',' << max_rel_err << ',' << rmse << '\n';
}

/**
 * @brief Check if the file is open.
 *
 * @return true
 * @return false
 */
bool CsvWriter::IsOpen() const { return file_.is_open(); }

/**
 * @brief Check if the file is in a good state.
 *
 * @return true
 * @return false
 */
bool CsvWriter::IsGood() const { return file_.good(); }

/**
 * @brief Flush the file buffer to disk.
 *
 */
void CsvWriter::Flush() { file_.flush(); }

/**
 * @brief Close the CSV file.
 *
 */
void CsvWriter::Close() {
    if (file_.is_open()) {
        file_.close();
    }
}

}  // namespace trt_allreduce::utils
