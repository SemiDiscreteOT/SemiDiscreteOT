#pragma once

/**
 * @brief Defines a set of inline string constants for ANSI color codes to be used in console output.
 */
namespace Color {
    inline const std::string reset   = "\033[0m";
    inline const std::string bold    = "\033[1m";
    inline const std::string red     = "\033[31m";
    inline const std::string green   = "\033[32m";
    inline const std::string yellow  = "\033[33m";
    inline const std::string blue    = "\033[34m";
    inline const std::string magenta = "\033[35m";
    inline const std::string cyan    = "\033[36m";
} 