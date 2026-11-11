#ifndef KNN2_TIMER_H
#define KNN2_TIMER_H
#include <chrono>
inline double elapsedTime() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif
