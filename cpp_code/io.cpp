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

int reverseInt(int i) {
    unsigned char ch;
    int rst = 0;
    for (int k = 0; k < 4; k++) {
        ch = (i >> (k * 8)) & 0xFF;
        rst += int(ch) << (8 * (3 - k));
    }
    return rst;
}

void readMnistImages(const std::string& filename, arma::mat& images) {
    std::ifstream file(filename, std::ifstream::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        int n_elems = 0;
        unsigned char* buf = NULL;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        n_elems = n_rows * n_cols;
        images.zeros(n_elems, number_of_images);
        buf = new unsigned char[n_elems * number_of_images];
        file.read((char*)buf, n_elems * number_of_images * sizeof(char));
        for (int j = 0; j < number_of_images; j++) {
            for (int i = 0; i < n_elems; i++) {
                images(i, j) = buf[i + j * n_elems] / 255.0;
            }
        }
        delete[] buf;
    } else throw std::runtime_error("File open failure.\n");
}

void readMnistLabels(const std::string& filename, arma::mat& labels) {
    std::ifstream file(filename, std::ifstream::binary);
    if (file.is_open()) {
        int magic_number = 0, number_of_images = 0;
        unsigned char* buf = NULL;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        labels.zeros(1, number_of_images);
        buf = new unsigned char[number_of_images];
        file.read((char*)buf, number_of_images);
        for (int i = 0; i < number_of_images; i++) {
            labels(0, i) = buf[i];
        }
        delete[] buf;
    } else throw std::runtime_error("File open failure.\n");
}

void readMnistImages(const std::string& filedir, const std::string& filename, arma::mat& images) {
    readMnistImages(filedir + SEPARATOR + filename, images);
}

void readMnistLabels(const std::string& filedir, const std::string& filename, arma::mat& labels) {
    readMnistLabels(filedir + SEPARATOR + filename, labels);
}
