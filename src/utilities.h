/**
 * @file utilities.h
 * @brief Provides helper functions for user input/output and filesystem
 * management.
 *
 * This header declares utility functions to handle user interaction via the
 * console and manage output directories for simulation results.
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <filesystem>
#include <string>

/**
 * @brief Clears the "solutions" directory.
 * * Removes all files within the "solutions" folder to ensure that output
 * from previous runs (e.g., .vtk files) does not mix with the current run.
 * Creates the directory if it does not exist.
 */
void clear_solutions_folder();

/**
 * @brief Prompts the user with a yes/no question.
 * * Accepts various inputs (y, yes, 1, s, si) for TRUE and (n, no, 0) for
 * FALSE. Loops until a valid input is provided.
 * * @param question The question string to display to the user.
 * @return true if the user answers affirmatively, false otherwise.
 */
bool ask_bool(const std::string &question);

/**
 * @brief Asks the user for a double precision number, offering a default value.
 * * If the user simply presses ENTER, the default value is returned.
 * Otherwise, the input is parsed as a double.
 * * @param question The prompt to display.
 * @param default_value The value to use if input is empty.
 * @return The user-provided double or the default value.
 */
double ask_double_default(const std::string &question,
                          const double       default_value);

/**
 * @brief Asks the user for an unsigned integer, offering a default value.
 * * Ensures the input is non-negative.
 * * @param question The prompt to display.
 * @param default_value The value to use if input is empty.
 * @return The user-provided unsigned int or the default value.
 */
unsigned int ask_uint_default(const std::string &question,
                              const unsigned int default_value);

#endif // UTILITIES_H