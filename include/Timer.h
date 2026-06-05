#pragma once

#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>

/**
 * @class Timer
 * @brief A utility class for measuring elapsed time with high precision.
 *
 * The Timer class provides functionality to track the passage of time
 * with methods to start, stop, and reset the timer. It can be used for
 * performance measurements and benchmarking in applications.
 *
 * This class is designed to abstract timing-related operations while ensuring
 * accurate results across different system clocks.
 */
class Timer final {
    struct Lap;

public:
    Timer() = default;

    ~Timer() {
        if (current_lap) {
            stop();
        }

        if (output_stream) {
            *output_stream << "# Total Duration: " << Lap::getDurationString(duration()) << '\n';
        } else {
#ifndef _DEBUG
            print();
#endif
        }
    }

    explicit Timer(std::ostream &os) : output_stream{&os} {
    }

    void start(const char *name) noexcept {
        current_lap = std::make_unique<Lap>(name);
    }

    void lap(const char *name) noexcept {
        if (current_lap) {
            stop();
        }
        start(name);
    }

    void stop() noexcept {
        if (current_lap) {
            current_lap->stop();

            if (output_stream) {
                *output_stream << " --> " << current_lap->toString() << '\n';
            }

            laps.push_back(*current_lap);
            current_lap.reset();
        }
    }

    void print(std::ostream &os = std::cout) const noexcept {
        os << "Timer Laps:\n";
        for (const auto &lap: laps) {
            os << " --> " << lap.toString() << '\n';
        }

        if (current_lap) {
            os << " --> Current Lap: " << current_lap->toString() << '\n';
        }

        os << "# Total Duration: " << Lap::getDurationString(duration()) << '\n';
    }

    [[nodiscard]] std::chrono::milliseconds duration() const noexcept {
        size_t total_ms = 0;
        for (const auto &lap: laps) {
            total_ms += std::chrono::duration_cast<std::chrono::milliseconds>(lap.end - lap.start).count();
        }

        return std::chrono::milliseconds(total_ms);
    }

private:
    void sortLaps() noexcept {
        std::ranges::sort(laps, [](const Lap &a, const Lap &b) noexcept {
            return a.end < b.end;
        });
    }

    struct Lap {
        static inline std::map<size_t, std::string> durations = {
            {1, "ms"},
            {1000, "s"},
            {60000, "min"},
            {3600000, "hr"},
            {86400000, "day"},
            {604800000, "week"},
            {2629743000, "month"},
        };

        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point end;
        const char *name;

        explicit Lap(const char *name) noexcept : name(name) {
            start = std::chrono::steady_clock::now();
        }

        void stop() noexcept {
            end = std::chrono::steady_clock::now();
        }

        static std::string getDurationString(const std::chrono::milliseconds &duration) noexcept {
            const auto durationCount = duration.count();
            if (durationCount < 0) {
                return "Invalid duration";
            }

            std::string result;

            for (const auto &[threshold, unit]: durations) {
                if (threshold > durationCount) {
                    break;
                }
                result = unit;
            }

            return std::to_string(durationCount) + " " + result;
        }

        [[nodiscard]] std::string toString() const noexcept {
            auto const label = std::string("Lap [") + name + "] ";
            auto const duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return label + getDurationString(duration);
        }
    };

    std::unique_ptr<Lap> current_lap;
    std::vector<Lap> laps;
    std::ostream *output_stream{};
};
