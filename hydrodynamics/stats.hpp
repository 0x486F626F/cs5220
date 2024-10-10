#include <fstream>
#include <vector>
#include <string>

struct stats {
    std::vector<std::vector<double>> t;
    std::vector<std::vector<std::string>> desc;

    static stats& get_stats() {
        static stats s;
        return s;
    }
    void print(std::string filename);
    void accu_time(const size_t i, const size_t j, const double d);
};
