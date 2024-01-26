CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

all: main decompressor

main: main.cpp
	$(CXX) $(CXXFLAGS) -o main main.cpp

decompressor: decompressor.cpp
	$(CXX) $(CXXFLAGS) -o decompress decompressor.cpp

clean:
	rm -f main decompress
