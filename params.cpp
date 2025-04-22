struct Params {
    int sample_int = 50;
    double sample_float = 1.5;
} Params;

void updateParams(int argc, char* argv[]) {
    for (int i = 1; i < argc; i += 2) {
        std::string key = argv[i];
        std::string value = argv[i + 1];
        std::istringstream iss(value);
        if (key == "sample_int") {
            iss >> Params.sample_int;
        }
        if (key == "sample_float") {
            iss >> Params.sample_float;
        }
    }
}
