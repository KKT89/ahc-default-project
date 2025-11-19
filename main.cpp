#include <bits/stdc++.h>
using namespace std;

template <typename T> struct dinic {
    struct edge {
        int to;
        T c, f;
    };
    T eps;
    const T inf = numeric_limits<T>::max();
    int n, m = 0;
    vector<edge> e;
    vector<vector<int>> g;
    vector<int> level, ptr;
    dinic(int n) : n(n), g(n), level(n), ptr(n) { eps = (T)1 / (T)1e9; }
    void add_edge(int s, int t, T c) {
        e.push_back({t, c, 0});
        e.push_back({s, 0, 0});
        g[s].push_back(m++);
        g[t].push_back(m++);
    }
    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        level[s] = 0;
        for (queue<int> q({s}); q.size(); q.pop()) {
            int s = q.front();
            for (int i : g[s]) {
                int t = e[i].to;
                if (level[t] == -1 and (e[i].c - e[i].f) > eps) {
                    level[t] = level[s] + 1;
                    q.push(t);
                }
            }
        }
        return (level[t] != -1);
    }
    T dfs(int s, int t, T psh) {
        if (!(psh > eps) or s == t) return psh;
        for (int &i = ptr[s]; i < (int)g[s].size(); ++i) {
            auto &eg = e[g[s][i]];
            if (level[eg.to] != level[s] + 1 or !(eg.c - eg.f > eps)) continue;
            T f = dfs(eg.to, t, min(psh, eg.c - eg.f));
            if (f > eps) {
                eg.f += f;
                e[g[s][i] ^ 1].f -= f;
                return f;
            }
        }
        return 0;
    }
    T max_flow(int s, int t) {
        T f = 0;
        while (bfs(s, t)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (1) {
                T c = dfs(s, t, inf);
                if (c > eps) {
                    f += c;
                } else {
                    break;
                }
            }
        }
        return f;
    }
    // ABC239-G
    vector<bool> min_cut(int s) {
        vector<bool> visited(n);
        queue<int> q;
        q.push(s);
        while (q.size()) {
            int p = q.front();
            q.pop();
            visited[p] = true;
            for (auto idx : g[p]) {
                auto eg = e[idx];
                if (eg.c - eg.f > eps and !visited[eg.to]) {
                    visited[eg.to] = true;
                    q.push(eg.to);
                }
            }
        }
        return visited;
    }
};

namespace timer {
constexpr double time_scale = 1.0;

// return in ms
int timer(bool reset = false) {
    // system_clock -> steady_clock に変更
    static auto st = chrono::steady_clock::now();
    if (reset) {
        st = chrono::steady_clock::now();
        return 0;
    } else {
        auto en = chrono::steady_clock::now();
        // millisecondsへのキャストを確実に行う
        int elapsed = (int)chrono::duration_cast<chrono::milliseconds>(en - st).count();
        return (int)round(elapsed / time_scale);
    }
}
} // namespace timer

struct mwc128xxa32 {
    using u32 = uint32_t;
    using u64 = uint64_t;

    // Default constructor: fixed seed initial state
    u32 x2 = 12345u;
    u32 x3 = 0xcafef00du;
    u64 c_x1 = (u64(0xd15ea5e5) << 32) | 23456u;
    mwc128xxa32() = default;

    // Internal constants: multiplier for MWC and reciprocal of 2^32 for [0,1) scaling
    static constexpr u32 MULT = 3487286589u;
    static constexpr double INV_2POW32 = 1.0 / 4294967296.0;

    // Generate next 32-bit random number
    __attribute__((always_inline)) inline u32 next() {
        u64 x = u64(x3) * MULT;
        u32 result = (x3 ^ x2) + (static_cast<u32>(c_x1) ^ static_cast<u32>(x >> 32));
        // shift state
        x3 = x2;
        x2 = static_cast<u32>(c_x1);
        c_x1 = x + (c_x1 >> 32);
        return result;
    }

    // Integer in [0, m)
    __attribute__((always_inline)) inline int nextInt(int m) { return int((u64(next()) * u64(m)) >> 32); }

    // Double in [0, 1)
    __attribute__((always_inline)) inline double nextDouble() { return next() * INV_2POW32; }

    // Log of uniform [0,1)
    inline double nextLog() { return std::log(nextDouble()); }
};
static thread_local mwc128xxa32 rnd32;

constexpr int NMAX = 21;
constexpr int KMAX = 401;
constexpr int INF = 0x3f3f3f3f;

int N, K, T_max;
vector<pair<int, int>> target;

const int dx[4] = {0, 0, 1, -1};
const int dy[4] = {1, -1, 0, 0};
const char dir[4] = {'R', 'L', 'D', 'U'};

bool graph[NMAX][NMAX][4];

struct RuleRow {
    int c, q, a, s, d;
};

struct Solution {
    int C{}, Q{};
    vector<vector<int>> init;
    vector<RuleRow> rules;
};

void input() {
    cin >> N >> K >> T_max;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j + 1 < N; ++j) {
            char c;
            cin >> c;
            graph[i][j][0] = graph[i][j + 1][1] = (c == '0');
        }
    }
    for (int i = 0; i + 1 < N; ++i) {
        for (int j = 0; j < N; ++j) {
            char c;
            cin >> c;
            graph[i][j][2] = graph[i + 1][j][3] = (c == '0');
        }
    }
    target.resize(K);
    for (int i = 0; i < K; ++i) {
        cin >> target[i].first >> target[i].second;
    }
}

void output(Solution &sol) {
    cout << sol.C << " " << sol.Q << " " << sol.rules.size() << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << sol.init[i][j] << (j + 1 == N ? '\n' : ' ');
        }
    }
    for (auto &r : sol.rules) {
        cout << r.c << " " << r.q << " " << r.a << " " << r.s << " " << dir[r.d] << "\n";
    }
}

// --- テンプレ・前処理 --- //

bool grid[NMAX][NMAX];

int dp[KMAX][NMAX][NMAX][5];
int goal_dist[KMAX][NMAX][NMAX][5];
using Pre = tuple<int, int, int, int>; // kid, x, y, dir
Pre pre[KMAX][NMAX][NMAX][5];

struct Node {
    int idx, x, y, dir;
};
Node que[KMAX * NMAX * NMAX * 5];

void calc_bfs() {
    int sx = target[0].first, sy = target[0].second;
    int gx = target[K - 1].first, gy = target[K - 1].second;
    int head = 0, tail = 0;
    auto in_bounds = [&](int x, int y) -> bool { return (0 <= x and x < N and 0 <= y and y < N); };
    auto valid_prev_idx = [&](int p_idx, int cur_idx, int cx, int cy) -> bool {
        if (p_idx < 0) return false;
        int expected = p_idx;
        if (p_idx != K - 1) {
            auto [tx, ty] = target[p_idx + 1];
            if (cx == tx and cy == ty) expected = p_idx + 1;
        }
        return expected == cur_idx;
    };
    auto push_state = [&](int p_idx, int px, int py, int pd, int ndist) -> void {
        if (p_idx < 0) return;
        if (!in_bounds(px, py)) return;
        if (goal_dist[p_idx][px][py][pd] != INF) return;
        goal_dist[p_idx][px][py][pd] = ndist;
        if (!(p_idx == 0 and px == sx and py == sy)) {
            que[tail++] = {p_idx, px, py, pd};
        }
    };
    for (int d = 0; d < 5; ++d) {
        goal_dist[K - 1][gx][gy][d] = 0;
        que[tail++] = {K - 1, gx, gy, d};
    }
    while (head < tail) {
        auto [idx, x, y, d] = que[head++];
        int cur_dist = goal_dist[idx][x][y][d];
        if (idx == 0 and x == sx and y == sy) continue;
        for (int delta = 0; delta <= 1; ++delta) {
            int p_idx = idx - delta;
            if (!valid_prev_idx(p_idx, idx, x, y)) continue;
            if (d != 4) {
                if (!grid[x][y]) continue;
                int px = x - dx[d], py = y - dy[d];
                if (in_bounds(px, py) and graph[px][py][d]) {
                    push_state(p_idx, px, py, d, cur_dist + 1);
                }
                for (int rd = 0; rd < 4; ++rd) {
                    int qx = x - dx[rd], qy = y - dy[rd];
                    if (!in_bounds(qx, qy) or !graph[qx][qy][rd]) continue;
                    push_state(p_idx, qx, qy, 4, cur_dist + 1);
                }
            } else {
                for (int rd = 0; rd < 4; ++rd) {
                    int px = x - dx[rd], py = y - dy[rd];
                    if (!in_bounds(px, py) or !graph[px][py][rd]) continue;
                    if (!grid[x][y]) {
                        if (grid[px][py]) {
                            push_state(p_idx, px, py, rd, cur_dist + 1);
                        }
                        push_state(p_idx, px, py, 4, cur_dist + 1);
                    }
                }
            }
        }
    }
}

struct DPState {
    int idx, x, y, d, dist, cost;
};

int buildDP() {
    memset(dp, 0x3f, sizeof(dp));
    memset(goal_dist, 0x3f, sizeof(goal_dist));
    calc_bfs();

    int sx = target[0].first, sy = target[0].second;
    dp[0][sx][sy][4] = 1;
    deque<DPState> dq;
    dq.push_back({0, sx, sy, 4, 0, 1});

    while (!dq.empty()) {
        auto [idx, x, y, d, dist, cost] = dq.front();
        dq.pop_front();
        if (dp[idx][x][y][d] < cost) continue;
        if (idx == K - 1) break;
        if (d != 4) {
            if (!graph[x][y][d]) continue;
            int nx = x + dx[d], ny = y + dy[d];
            int n_idx = idx + (idx != K - 1 and nx == target[idx + 1].first and ny == target[idx + 1].second ? 1 : 0);
            int nd = (!grid[nx][ny] ? 4 : d);
            if (dist + 1 + goal_dist[n_idx][nx][ny][nd] > T_max) continue;
            int add = (!grid[nx][ny] and n_idx != K - 1 ? 1 : 0);
            int n_cost = cost + add;
            if (dp[n_idx][nx][ny][nd] > n_cost) {
                dp[n_idx][nx][ny][nd] = n_cost;
                pre[n_idx][nx][ny][nd] = {idx, x, y, d};
                if (add == 0) {
                    dq.push_front({n_idx, nx, ny, nd, dist + 1, n_cost});
                } else {
                    dq.push_back({n_idx, nx, ny, nd, dist + 1, n_cost});
                }
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                if (!graph[x][y][i]) continue;
                int nx = x + dx[i], ny = y + dy[i];
                int n_idx = idx + (idx != K - 1 and nx == target[idx + 1].first and ny == target[idx + 1].second ? 1 : 0);
                int add = (!grid[nx][ny] and n_idx != K - 1 ? 1 : 0);
                int n_cost = cost + add;

                if (!grid[nx][ny]) {
                    int nd = 4;
                    if (dist + 1 + goal_dist[n_idx][nx][ny][nd] > T_max) continue;
                    if (dp[n_idx][nx][ny][nd] > n_cost) {
                        dp[n_idx][nx][ny][nd] = n_cost;
                        pre[n_idx][nx][ny][nd] = {idx, x, y, d};
                        if (add == 0) {
                            dq.push_front({n_idx, nx, ny, nd, dist + 1, n_cost});
                        } else {
                            dq.push_back({n_idx, nx, ny, nd, dist + 1, n_cost});
                        }
                    }
                } else {
                    for (int nd = 0; nd < 4; ++nd) {
                        if (dist + 1 + goal_dist[n_idx][nx][ny][nd] > T_max) continue;
                        if (dp[n_idx][nx][ny][nd] > n_cost) {
                            dp[n_idx][nx][ny][nd] = n_cost;
                            pre[n_idx][nx][ny][nd] = {idx, x, y, d};
                            if (add == 0) {
                                dq.push_front({n_idx, nx, ny, nd, dist + 1, n_cost});
                            } else {
                                dq.push_back({n_idx, nx, ny, nd, dist + 1, n_cost});
                            }
                        }
                    }
                }
            }
        }
    }

    int gx = target[K - 1].first, gy = target[K - 1].second;
    int res = INF;
    for (int i = 0; i < 5; ++i) {
        res = min(res, dp[K - 1][gx][gy][i]);
    }
    if (res == INF) return -1;
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    constexpr bool DEBUG = false;
    constexpr double TIME_LIMIT = 180;
    timer::timer(true);
    input();

    Solution best;
    best.C = 1e9, best.Q = 1e9;

    // DP更新済みを仮定
    auto resolve_from_dp = [&]() -> vector<pair<int, int>> {
        int sx = target[0].first, sy = target[0].second;
        int gx = target[K - 1].first, gy = target[K - 1].second;
        vector<pair<int, int>> route;
        int g_idx = K - 1;
        int best_d = 0;
        for (int i = 0; i < 5; ++i) {
            if (dp[K - 1][gx][gy][i] < dp[K - 1][gx][gy][best_d]) {
                best_d = i;
            }
        }
        route.reserve(goal_dist[0][sx][sy][4] + 1);

        while (gx != sx || gy != sy || g_idx != 0) {
            auto [idx, x, y, d] = pre[g_idx][gx][gy][best_d];
            route.emplace_back(gx, gy);
            gx = x, gy = y, g_idx = idx;
            best_d = d;
        }
        route.emplace_back(sx, sy);
        return route;
    };

    // 逆順のrouteを渡す
    auto update_sol = [&](const vector<pair<int, int>> &route) -> void {
        cerr << "route size: " << route.size() << "\n";

        auto resolve_dir = [&](int x, int y, int nx, int ny) -> int {
            if (x + dx[0] == nx and y + dy[0] == ny) return 0;
            else if (x + dx[1] == nx and y + dy[1] == ny) return 1;
            else if (x + dx[2] == nx and y + dy[2] == ny) return 2;
            else if (x + dx[3] == nx and y + dy[3] == ny) return 3;
            assert(false);
        };

        // (x, y, d, dd)
        vector<tuple<int, int, int, int>> op_info;
        vector<vector<bool>> can_skip(N, vector<bool>(N, true));

        if (DEBUG) {
            for (auto &t : route) {
                cerr << "route: (" << t.first << "," << t.second << ")\n";
            }
        }

        int slide_dir = -1;
        for (int i = 1; i < (int)route.size(); ++i) {
            auto [x, y] = route[i];
            auto [nx, ny] = route[i - 1];
            int dir = resolve_dir(x, y, nx, ny);
            if (grid[x][y] and i + 1 < (int)route.size()) {
                slide_dir = dir;
            } else {
                if (i == 1) slide_dir = dir;

                bool skip = can_skip[x][y];
                skip &= (dir == slide_dir); // slide_dir -1 もパスできる
                if (i + 1 < (int)route.size()) {
                    auto [px, py] = route[i + 1];
                    int p_dir = resolve_dir(px, py, x, y);
                    skip &= (dir == p_dir);
                } else {
                    skip = false;
                }

                if (skip) continue;
                can_skip[x][y] = false;
                op_info.emplace_back(x, y, dir, slide_dir);
                slide_dir = -1;
            }
        }

        if (DEBUG) {
            for (auto &t : op_info) {
                auto [x, y, d, dd] = t;
                cerr << "op_info: (" << x << "," << y << ") d=" << dir[d] << " dd=" << (dd == -1 ? -1 : dir[dd]) << "\n";
            }
        }

        // C/Q を決定する
        vector<int> count(5), has(4);
        count[0] += 1;
        for (int i = 1; i < (int)op_info.size(); ++i) {
            int d = get<3>(op_info[i]);
            if (d == -1) count[4] += 1;
            else count[d] += 1;
        }

        int best_val = 1e9;
        int C = -1, Q = -1;
        for (int c = 1; c <= 100; ++c) {
            int rem = count[4];
            vector<int> tmp_has(4);
            for (int i = 0; i < 4; ++i) {
                tmp_has[i] = (count[i] + c - 1) / c;
                int md = count[i] % c;
                if (md != 0) md = (c - md);
                int val = min(rem, md);
                rem -= val;
            }
            tmp_has[0] += (rem + c - 1) / c;
            int q = tmp_has[0] + tmp_has[1] + tmp_has[2] + tmp_has[3];
            if (c + q + 1 < best_val) {
                best_val = c + 1 + q;
                C = c + 1;
                Q = q;
                has = tmp_has;
            }
        }

        //        if (C + Q >= best.C + best.Q) return;

        C = 200;
        auto slv = [&](vector<int> has) -> bool {
            vector<int> can_use = {has[0] * (C - 1) - count[0], has[1] * (C - 1) - count[1], has[2] * (C - 1) - count[2], has[3] * (C - 1) - count[3]};

            Solution sol{};
            sol.C = C, sol.Q = Q;
            sol.init.assign(N, vector<int>(N, 0));
            vector<RuleRow> res;
            res.reserve(Q + (int)op_info.size());

            array<int, 4> off{0, 0, 0, 0};
            for (int d = 1; d < 4; ++d) {
                off[d] = off[d - 1] + has[d - 1];
            }

            // 後ろから埋めていく
            vector<int> used(Q, 1);
            int nxt_q;
            int mxC = 0;
            for (int i = 0; i < (int)op_info.size(); ++i) {
                auto [x, y, d, dd] = op_info[i];
                int tc = sol.init[x][y], tq, td = d;

                if (i == 0) {
                    tq = off[dd];
                } else if (dd == -1) {
                    tq = nxt_q;
                } else {
                    auto [px, py, pd, pdd] = op_info[i - 1];
                    int dif = abs(x + dx[d] - px) + abs(y + dy[d] - py);
                    for (int j = 0; j < dif; ++j) {
                        if (nxt_q == off[dd]) {
                            nxt_q = off[dd] + has[dd] - 1;
                        } else {
                            nxt_q -= 1;
                        }
                    }
                    tq = nxt_q;
                }

                bool find = false;
                int qv, cur_c, cur_q;
                if (i + 1 == op_info.size()) {
                    cur_q = 0;
                    cur_c = used[cur_q]++;
                } else {
                    auto [nx, ny, nd, ndd] = op_info[i + 1];
                    qv = ndd;
                    if (qv == -1) {
                        //                    for (int j = 0; j < 4; ++j) {
                        //                        if (can_use[j] > 0) {
                        //                            can_use[j] -= 1;
                        //                            qv = j;
                        //                            break;
                        //                        }
                        //                    }
                        //                    assert(qv != -1);
                        for (auto &[c, q, a, s, ddd] : res) {
                            if (tc == a and tq == s and td == ddd) {
                                cur_q = q;
                                cur_c = c;
                                find = true;
                                break;
                            }
                        }
                    } else {
                        int l = off[qv], r = off[qv] + has[qv];
                        for (auto &[c, q, a, s, ddd] : res) {
                            if (tc == a and tq == s and td == ddd) {
                                if (l <= q and q < r) {
                                    cur_q = q;
                                    cur_c = c;
                                    find = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (!find) {
                        if (qv == -1) {
                            qv = 0;
                            for (int j = 0; j < 4; ++j) {
                                if (can_use[j] >= can_use[qv]) {
                                    qv = j;
                                }
                            }
                            assert(can_use[qv] > 0);
                            can_use[qv] -= 1;
                        }
                        {
                            int l = off[qv], r = off[qv] + has[qv];
                            cur_q = l;
                            for (int j = l; j < r; ++j) {
                                if (used[j] <= used[cur_q]) {
                                    cur_q = j;
                                }
                            }
                        }
                        cur_c = used[cur_q]++;
                    }
                }

                if (!find) res.push_back({cur_c, cur_q, tc, tq, td});
                sol.init[x][y] = cur_c;
                nxt_q = cur_q;
                mxC = max(mxC, cur_c + 1);
            }

            // 色0
            for (int d = 0; d < 4; ++d) {
                int l = off[d], r = off[d] + has[d];
                for (int j = l; j + 1 < r; ++j) {
                    res.push_back({0, j, 0, j + 1, d});
                }
                res.push_back({0, r - 1, 0, l, d});
            }

            sol.C = mxC;
            sol.rules = std::move(res);
            if (best.C + best.Q > sol.C + sol.Q) {
                best = sol;
                return true;
            } else {
                return false;
            }
        };

        slv(has);
    };

    if (1) {
        using P = pair<int, int>;
        using P3 = tuple<int, int, int>;
        using P5 = pair<P, P3>;
        // dist, 曲がり回数
        vector<vector<vector<P>>> dp1(KMAX, vector<vector<P>>(N, vector<P>(N, {1e9, -1})));
        vector<vector<vector<P3>>> pre1(KMAX, vector<vector<P3>>(N, vector<P3>(N, {-1, -1, -1})));

        priority_queue<P5, vector<P5>, greater<>> pq;
        int sx = target[0].first, sy = target[0].second;
        for (int i = 0; i < 4; ++i) {
            if (!graph[sx][sy][i]) continue;
            int nx = sx + dx[i], ny = sy + dy[i];
            int n_idx = (nx == target[1].first and ny == target[1].second ? 1 : 0);
            dp1[n_idx][nx][ny] = {1, 0};
            pre1[n_idx][nx][ny] = {0, sx, sy};
            pq.push({dp1[n_idx][nx][ny], {n_idx, nx, ny}});
        }

        while (!pq.empty()) {
            auto [p, q] = pq.top();
            pq.pop();
            int dist = p.first, turns = p.second;
            auto [idx, x, y] = q;
            if (dp1[idx][x][y] < p) continue;
            if (idx == K - 1) break;
            auto [p_idx, px, py] = pre1[idx][x][y];
            for (int d = 0; d < 4; ++d) {
                if (!graph[x][y][d]) continue;
                int nx = x + dx[d], ny = y + dy[d];
                int n_idx = idx + (nx == target[idx + 1].first and ny == target[idx + 1].second ? 1 : 0);
                int n_dist = dist + 1, n_turn = turns + (nx != x + (x - px) or ny != y + (y - py));
                if (dp1[n_idx][nx][ny] > P{n_dist, n_turn}) {
                    dp1[n_idx][nx][ny] = {n_dist, n_turn};
                    pre1[n_idx][nx][ny] = {idx, x, y};
                    pq.push({dp1[n_idx][nx][ny], {n_idx, nx, ny}});
                }
            }
        }
        int idx = K - 1;
        auto [gx, gy] = target[K - 1];
        auto res = dp1[K - 1][gx][gy];
        cerr << "initial dist/turns: " << res.first << " " << res.second << "\n";

        vector<P> path;
        path.emplace_back(gx, gy);
        while (idx != 0 or gx != sx or gy != sy) {
            auto [p_idx, px, py] = pre1[idx][gx][gy];
            gx = px, gy = py, idx = p_idx;
            path.emplace_back(gx, gy);
        }
        reverse(path.begin(), path.end());

        //        for (auto [x, y] : path) {
        //            cerr << "(" << x << "," << y << ") " << "\n";
        //        }

        vector<pair<P, P>> v;
        for (int i = 1; i + 1 < path.size(); ++i) {
            auto [x, y] = path[i];
            auto [px, py] = path[i - 1];
            auto [nx, ny] = path[i + 1];
            if (x - px == nx - x and y - py == ny - y) continue;
            v.push_back({path[i - 1], path[i]});
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                grid[i][j] = true;
            }
        }

        vector<vector<int>> idL(N, vector<int>(N, -1));
        vector<vector<int>> idR(N, vector<int>(N, -1));
        int L = 0, R = 0;

        // まず、v に出てくるマスのうち、grid[x][y]==true のものだけ番号を振る
        for (auto [prev, cur] : v) {
            auto [x1, y1] = prev;
            auto [x2, y2] = cur;
            if (!grid[x1][y1] || !grid[x2][y2]) continue;

            if ((x1 + y1) % 2 == 0) {
                if (idL[x1][y1] == -1) idL[x1][y1] = L++;
                if (idR[x2][y2] == -1) idR[x2][y2] = R++;
            } else {
                if (idL[x2][y2] == -1) idL[x2][y2] = L++;
                if (idR[x1][y1] == -1) idR[x1][y1] = R++;
            }
        }

        int S = L + R;
        int T = L + R + 1;
        dinic<int> mf(T + 1);

        // S -> 左
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                int id = idL[x][y];
                if (id == -1) continue;
                mf.add_edge(S, id, 1);
            }
        }

        // 右 -> T
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                int id = idR[x][y];
                if (id == -1) continue;
                mf.add_edge(L + id, T, 1);
            }
        }

        // 制約から辺を張る
        for (auto [prev, cur] : v) {
            auto [x1, y1] = prev;
            auto [x2, y2] = cur;
            if (!grid[x1][y1] || !grid[x2][y2]) continue;

            if ((x1 + y1) % 2 == 0) {
                int u = idL[x1][y1];
                int v2 = idR[x2][y2];
                if (u != -1 && v2 != -1) mf.add_edge(u, L + v2, 1);
            } else {
                int u = idL[x2][y2];
                int v2 = idR[x1][y1];
                if (u != -1 && v2 != -1) mf.add_edge(u, L + v2, 1);
            }
        }

        mf.max_flow(S, T);
        auto vis = mf.min_cut(S); // S から残余グラフで到達可能な頂点

        // 左側
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                int id = idL[x][y];
                if (id == -1) continue;
                if (!vis[id]) { // 頂点被覆に含まれる
                    grid[x][y] = false;
                }
            }
        }

        // 右側
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                int id = idR[x][y];
                if (id == -1) continue;
                if (vis[L + id]) { // 頂点被覆に含まれる
                    grid[x][y] = false;
                }
            }
        }

        for (auto [prev, cur] : v) {
            auto [x1, y1] = prev;
            auto [x2, y2] = cur;
            if (grid[x1][y1] and grid[x2][y2]) assert(false);
        }
    }

    vector<pair<int, int>> kouho;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!grid[i][j]) {
                kouho.emplace_back(i, j);
            }
        }
    }

    // 初期解
    int cur_score = buildDP();
    assert(cur_score != -1);
    auto route = resolve_from_dp();
    update_sol(route);

    int try_cnt = 0;

    while (true) {
        int x = rnd32.nextInt(N), y = rnd32.nextInt(N);
        if (grid[x][y]) {
            if (graph[x][y][0] and !grid[x + dx[0]][y + dy[0]]) continue;
            if (graph[x][y][1] and !grid[x + dx[1]][y + dy[1]]) continue;
            if (graph[x][y][2] and !grid[x + dx[2]][y + dy[2]]) continue;
            if (graph[x][y][3] and !grid[x + dx[3]][y + dy[3]]) continue;
        }
        if (timer::timer() > TIME_LIMIT) break;
        grid[x][y] = !grid[x][y];
        try_cnt += 1;
        int score = buildDP();
        if (score == -1 or score > cur_score) {
            grid[x][y] = !grid[x][y];
            continue;
        }
        cur_score = score;
    }
    buildDP();
    route = resolve_from_dp();
    update_sol(route);

    output(best);
    cerr << "try_cnt: " << try_cnt << "\n";
    cerr << "best (C-1)Q: " << (best.C - 1) * best.Q << "\n";
}
