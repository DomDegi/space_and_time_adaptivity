#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <filesystem>

// Clears the "solutions" folder to avoid mixing old VTK files
void clear_solutions_folder();

// Asks a yes/no question to the user via console
bool ask_bool(const std::string &question);

// Asks for a double value, providing a default if the user just presses Enter
double ask_double_default(const std::string &question, const double default_value);

// Asks for an unsigned int value, providing a default if the user just presses Enter
unsigned int ask_uint_default(const std::string &question, const unsigned int default_value);

#endif // UTILITIES_H
