#include "log-utils.h"

std::string os_name()
{
    #ifdef _WIN64
    return "Windows 64-bit";
    #elif _WIN32
    return "Windows 32-bit";
    #elif __APPLE__ || __MACH__
    return "Mac OSX";
    #elif __linux__
    return "Linux";
    #elif __FreeBSD__
    return "FreeBSD";
    #elif __unix || __unix__
    return "Unix";
    #else
    return "Other";
    #endif
}

std::shared_ptr<spdlog::logger> _global_logger; // thread-safe logger

void init_logging(std::string name, std::string greeting_text) {
//    // Create a file rotating logger with 10mb size max and 30 rotated files
//    auto max_size_in_bytes = 1024 * 1024 * 100;
//    auto max_files = 100;
//    auto file_logger = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/log-" + name + ".txt", max_size_in_bytes, max_files);
//    file_logger->set_pattern("[pid: %P, tid: %t] [%Y-%m-%d-%H:%M:%S.%e] [%^%l%$] %v");
//	file_logger->set_level(spdlog::level::debug);

	auto console_logger = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_logger->set_level(spdlog::level::info);
    console_logger->set_pattern("%v");

    auto logger = std::make_shared<spdlog::logger>(name, spdlog::sinks_init_list{console_logger});
    logger->set_level(spdlog::level::debug);
    logger->info(greeting_text);
    _global_logger = logger;

//    auto logger = std::make_shared<spdlog::logger>(name, spdlog::sinks_init_list{file_logger, console_logger});
//    logger->set_level(spdlog::level::debug);
//    logger->info(greeting_text);
//    _global_logger = logger;

    LOG_INFO("logging initialized.");
    LOG_INFO("running on OS: " + os_name());

}