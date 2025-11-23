#pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;

constexpr bool DEBUG = false;
constexpr double TIME_LIMIT = 1800;

namespace timer {
constexpr double time_scale = 1.0;
// return in ms
int timer(bool reset = false) {
    static auto st = chrono::steady_clock::now();
    if (reset) {
        st = chrono::steady_clock::now();
        return 0;
    } else {
        auto en = chrono::steady_clock::now();
        int elapsed = (int)chrono::duration_cast<chrono::milliseconds>(en - st).count();
        return (int)round(elapsed / time_scale);
    }
}
} // namespace timer

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    constexpr bool DEBUG = false;
    constexpr double TIME_LIMIT = 1800;
    timer::timer(true);
}
