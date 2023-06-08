#include <string>
#include <locale>
#include <algorithm>

#include "string-utils.h"

/**
 * @brief Determines whether a string ends with the specified ending.
 * @param str the string to ananlyze
 * @param ending the ending
 */
bool ends_with (std::string const &str, std::string const &ending) {
    if (str.length() >= ending.length()) {
        return (0 == str.compare (str.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

/**
 * @brief Returns the uppercase version of the specified string.
 * @param the string to convert
 * @return the uppercase version of the specified string
 */
std::string to_upper(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), 
                   [](unsigned char c){ return std::toupper(c); }
                  );
    return str;
}

/**
 * @brief Returns the lowercase version of the specified string.
 * @param the string to convert
 * @return the lowercase version of the specified string
 */
std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), 
                   [](unsigned char c){ return std::tolower(c); }
                  );
    return str;
}

/**
 * @brief Appends the specified ending to the string if it does not already end with the specified ending (case agnostic).
 * @param str string to analyze
 * @param ending the ending
 * @return a copy of the string with the specified ending (converted to lowercase) or an unmodified copy of the string if it already ends with the specified ending
 */
std::string file_name_with_ending(std::string const &str, std::string const &ending) {
    auto ending_lower = to_lower(std::string(ending));
    if(!ends_with(to_lower(std::string(str)), ending_lower)) {
        return str + ending_lower;
    } else {
        return str;
    }
}