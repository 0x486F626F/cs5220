#include "stats.hpp"

void stats::print(std::string filename) {
    std::ofstream fout(filename, std::ios::app);
    double sum = 0;
    for (size_t i = 0; i < t.size(); i ++) 
        sum += t[i][0];
    fout << sum << std::endl;
    for (size_t i = 0; i < t.size(); i ++) {
        fout << "\t" << t[i][0] << std::endl;
        for (size_t j = 1; j < t[i].size(); j ++)
            if (t[i][j] > 100000) 
            fout << "\t\t" << int(t[i][j]) << std::endl;
            else
            fout << "\t\t" << t[i][j] << std::endl;
    }
    fout << "==========" << std::endl;
}


void stats::accu_time(const size_t i, const size_t j, const double d) {
    while (t.size() <= i) t.push_back(std::vector<double>());
    while (t[i].size() <= j) t[i].push_back(0);
    t[i][j] += d;
}
