#pragma once

#include <string>
#include <memory>

#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

extern std::shared_ptr<spdlog::logger> _global_logger;

void init_logging(std::string name, std::string greeting_text = "<no greeting text specified>");

#define LOG_INFO(first, ...)  _global_logger->info( "> " + std::string(first), ##__VA_ARGS__);
#define LOG_WARN(first, ...) _global_logger->warn("> [WARNING]: " + std::string(first), ##__VA_ARGS__);
#define LOG_ERROR(first, ...) _global_logger->error("> [ERROR]: " + std::string(first), ##__VA_ARGS__);
#define LOG_DEBUG(first, ...) _global_logger->debug("> " + std::string(first), ##__VA_ARGS__);

