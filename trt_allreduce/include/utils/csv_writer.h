/**
 * @file csv_writer.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Lightweight CSV writer.
 * @version 0.1
 * @date 2025-07-04
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <fstream>
#include <string>

namespace trt_allreduce::utils {

/**
 * @brief Lightweight CSV writer.
 *
 */
class CsvWriter {
   private:
    std::ofstream file_;

   public:
    /**
     * @brief Construct a new Csv Writer object
     *
     * @param path Path to the CSV file.
     * @param with_header Whether to write the header row.
     */
    CsvWriter(const std::string& path, bool with_header = true);

    /**
     * @brief Destroy the Csv Writer object
     *
     */
    ~CsvWriter();

    /**
     * @brief Write a row to the CSV file.
     *
     * @param impl Implementation name.
     * @param bytes Number of bytes in the message.
     * @param latency_ms Latency in milliseconds.
     * @param bandwidth_tb Bandwidth in TBps.
     * @param accuracy_pct Accuracy percentage.
     * @param max_abs_err Maximum absolute error.
     * @param mean_abs_err Mean absolute error.
     * @param max_rel_err Maximum relative error.
     * @param rmse Root mean square error.
     */
    void WriteRow(const std::string& impl, std::size_t bytes, double latency_ms, double bandwidth_tb, double accuracy_pct, float max_abs_err, double mean_abs_err,
                  float max_rel_err, double rmse);

    /**
     * @brief Check if the file is open.
     *
     * @return true
     * @return false
     */
    bool IsOpen() const;

    /**
     * @brief Check if the file is in a good state.
     *
     * @return true
     * @return false
     */
    bool IsGood() const;

    /**
     * @brief Flush the file buffer to disk.
     *
     */
    void Flush();

    /**
     * @brief Close the CSV file.
     *
     */
    void Close();
};

}  // namespace trt_allreduce::utils
