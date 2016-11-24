#include "io.h"
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <algorithm>

const char SEPARATOR = '/';
const char WIDTH = 20;

bool writeSGDProfile(const std::string& filedir, const std::string& filename, const SGDProfile& sgdProfile) {
    std::string filePath = filedir + SEPARATOR + filename;
    std::ofstream ofs;
    int n = sgdProfile.times.size();
    double timeCost = 0;
    ofs.open(filePath, std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open()) {
        ofs.setf(std::ios::scientific, std::ios::floatfield);
        ofs << "Period: " << sgdProfile.T << std::endl;
        ofs << std::setw(WIDTH) << "Epoch"  << std::setw(WIDTH) << "Time cost" << std::setw(WIDTH) << "Loss" << std::endl;  
        for (int i = 0; i < n; i++) {
            if ((i + 1) % sgdProfile.T == 0) {
                timeCost = std::accumulate(sgdProfile.times.begin() + i - sgdProfile.T, sgdProfile.times.begin() + i, 0.0);
                ofs << std::setw(WIDTH) << i + 1 << std::setw(WIDTH) << timeCost << std::setw(WIDTH) << sgdProfile.objs[i / sgdProfile.T] << std::endl;
            }
        }
    } else throw std::runtime_error("File open failure.\n");
    ofs.close();
    return true;
}

