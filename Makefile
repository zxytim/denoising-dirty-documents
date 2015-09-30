all: images2csv

images2csv: images2csv.cc
	$(CXX) images2csv.cc -o images2csv -std=c++11 -O3 $$(pkg-config --libs --cflags opencv)
