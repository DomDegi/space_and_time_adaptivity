/**
 * @file utilities.cc
 * @brief Implementation of helper functions defined in utilities.h.
 */

#include "utilities.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

void clear_solutions_folder() {
    // Ensure the directory exists
    std::filesystem::create_directories("solutions");

    // Iterate over directory entries and remove them
    for (const auto &entry : std::filesystem::directory_iterator("solutions"))
        std::filesystem::remove_all(entry.path());
}

bool ask_bool(const std::string &question) {
    while (true) {
        std::cout << question << " (y/n): ";
        std::string input;
        std::cin >> input;

        // Normalize input to lowercase for case-insensitive comparison
        std::transform(input.begin(), input.end(), input.begin(), ::tolower);

        // Check for affirmative answers (English and Italian common usage)
        if (input == "y" || input == "yes" || input == "1" || input == "s" ||
            input == "si")
            return true;

        // Check for negative answers
        if (input == "n" || input == "no" || input == "0")
            return false;

        std::cout << "Invalid input. Please answer 'y' (yes) or 'n' (no).\n";
    }
}

double ask_double_default(const std::string &question,
                          const double default_value) {
    while (true) {
        std::cout << question << " [default=" << default_value << "]: ";
        std::string line;
        // Read the full line to handle empty input (just Enter)
        std::getline(std::cin >> std::ws, line);

        if (line.empty())
            return default_value;

        std::stringstream ss(line);
        double v;
        // Check if parsing to double is successful and entire input was
        // consumed
        if ((ss >> v) && (ss.eof() || ss.peek() == EOF))
            return v;

        std::cout << "Invalid input. Please enter a number.\n";
    }
}

unsigned int ask_uint_default(const std::string &question,
                              const unsigned int default_value) {
    while (true) {
        std::cout << question << " [default=" << default_value << "]: ";
        std::string line;
        std::getline(std::cin >> std::ws, line);

        if (line.empty())
            return default_value;

        std::stringstream ss(line);
        int v;
        // Parse as int first to check for negative numbers, then cast
        // Also verify entire input was consumed
        if ((ss >> v) && v >= 0 && (ss.eof() || ss.peek() == EOF))
            return static_cast<unsigned int>(v);

        std::cout << "Invalid input. Please enter a positive integer (or 0).\n";
    }
}