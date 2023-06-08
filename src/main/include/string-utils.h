#pragma once

#include <string>

/**
 * @brief Determines whether a string ends with the specified ending.
 * @param str the string to ananlyze
 * @param ending the ending
 */
bool ends_with (std::string const &str, std::string const &ending);

/**
 * @brief Returns the uppercase version of the specified string.
 * @param the string to convert
 * @return the uppercase version of the specified string
 */
std::string to_upper(std::string const str);

/**
 * @brief Returns the lowercase version of the specified string.
 * @param the string to convert
 * @return the lowercase version of the specified string
 */
std::string to_lower(std::string const str);

/**
 * @brief Appends the specified ending to the string if it does not already end with the specified ending (case agnostic).
 * @param str string to analyze
 * @param ending the ending
 * @return a copy of the string with the specified ending or an unmodified copy of the string if it already ends with the specified ending
 */
std::string file_name_with_ending(std::string const &str, std::string const &ending);

