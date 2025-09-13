struct TimeKeeperDouble {
  private:
    chrono::high_resolution_clock::time_point startTime;
    double timeThreshold = 0;
    double nowTime = 0;

  public:
    explicit TimeKeeperDouble(const double time_threshold) : startTime(chrono::high_resolution_clock::now()), timeThreshold(time_threshold) {}
    void setNowTime() {
        auto diff = chrono::high_resolution_clock::now() - this->startTime;
        this->nowTime = chrono::duration_cast<std::chrono::microseconds>(diff).count() * 1e-3; // ms
    }
    [[nodiscard]] double getNowTime() const { return this->nowTime; }
    [[nodiscard]] bool isTimeOver() const { return nowTime >= timeThreshold; }
};