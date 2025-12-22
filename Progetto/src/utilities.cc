#include "utilities.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

void clear_solutions_folder()
{
  std::filesystem::create_directories("solutions");
  for (const auto &entry : std::filesystem::directory_iterator("solutions"))
    std::filesystem::remove_all(entry.path());
}

bool ask_bool(const std::string &question)
{
  while (true)
  {
    std::cout << question << " (y/n): ";
    std::string input;
    std::cin >> input;

    std::transform(input.begin(), input.end(), input.begin(), ::tolower);

    if (input == "y" || input == "yes" || input == "1" || input == "s" || input == "si")
      return true;

    if (input == "n" || input == "no" || input == "0")
      return false;

    std::cout << "Invalid input. Please answer 'y' (yes) or 'n' (no).\n";
  }
}

double ask_double_default(const std::string &question, const double default_value)
{
  while (true)
  {
    std::cout << question << " [default=" << default_value << "]: ";
    std::string line;
    std::getline(std::cin >> std::ws, line);

    if (line.empty())
      return default_value;

    std::stringstream ss(line);
    double v;
    if (ss >> v)
      return v;

    std::cout << "Invalid input. Please enter a number.\n";
  }
}

unsigned int ask_uint_default(const std::string &question, const unsigned int default_value)
{
  while (true)
  {
    std::cout << question << " [default=" << default_value << "]: ";
    std::string line;
    std::getline(std::cin >> std::ws, line);

    if (line.empty())
      return default_value;

    std::stringstream ss(line);
    int v;
    if ((ss >> v) && v >= 0)
      return static_cast<unsigned int>(v);

    std::cout << "Invalid input. Please enter a positive integer (or 0).\n";
  }
}
