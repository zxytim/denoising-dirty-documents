/*
 * $File: images2csv.cc
 * $Date: Wed Sep 30 11:35:22 2015 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include <opencv2/opencv.hpp>
#include <fstream>
#include <cstdio>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>


std::vector<std::string> ls(const std::string &dirname) {
	DIR *dir;
	struct dirent *ent;
	std::vector<std::string> ret;
	if ((dir = opendir(dirname.c_str())) != NULL) {
		while ((ent = readdir (dir)) != NULL) {
			std::string fname = ent->d_name;
			if (fname == "." || fname == "..")
				continue;
			ret.emplace_back(std::move(fname));
		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");
	}
	return ret;
}


struct ImageItem {
	std::string dirname, fname, fpath;
	int id;
	cv::Mat img;

	ImageItem(std::string dirname, std::string fname):
		dirname(dirname), fname(fname),
		fpath(dirname + "/" + fname),
		id(0),
		img(cv::imread(fpath, cv::IMREAD_GRAYSCALE))
	{
		for (size_t i = 0; i < fname.size(); i ++) {
			if (!std::isdigit(fname[i]))
				break;
			id = id * 10 + fname[i] - '0';
		}
	}


};

void write_csv(const std::string &output_path, const std::vector<ImageItem> &items) {
	FILE *fout = fopen(output_path.c_str(), "wb");

	fprintf(fout, "id,value\n");
	for (auto &item: items) {
		auto img = item.img;
		for (auto i = 0; i < img.rows; i ++)
			for (auto j = 0; j < img.cols; j ++) {
				fprintf(fout, "%d_%d_%d,%f\n", item.id, i, j, img.at<uchar>(i, j) / 255.0f);
			}
	}

	fclose(fout);



}


int main(int argc, char *argv[]) {
	if (argc != 3) {
		printf("Usage: %s <image_dir> <out_csv>\n", argv[0]);
		exit(1);
	}

	auto image_dir = std::string(argv[1]),
		 output_path = std::string(argv[2]);

//    std::sort(std::begin(image_paths), std::end(image_paths),
//            [](const std::string &a, const std::string &b){
//            return a < b;
//            });


	auto image_fnames = ls(image_dir);
	std::vector<ImageItem> items;
	for (auto &fname: image_fnames)
		items.emplace_back(image_dir, fname);

	write_csv(output_path, items);


	return 0;
}

/*
 * vim: syntax=cpp11.doxygen foldmethod=marker foldmarker=f{{{,f}}}
 */

