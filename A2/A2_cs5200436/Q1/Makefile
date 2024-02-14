CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

SRCS = FormatFiles/FSG_format.cpp FormatFiles/Gaston_format.cpp FormatFiles/Gspan_format.cpp
OBJS = $(SRCS:.cpp=.o)
EXECUTABLES = FormatFiles/FSG_format FormatFiles/Gaston_format FormatFiles/Gspan_format

all: $(EXECUTABLES)

FSG_format: FSG_format.o
	$(CXX) $(CXXFLAGS) -o $@ $^

Gaston_format: Gaston_format.o
	$(CXX) $(CXXFLAGS) -o $@ $^

Gspanformat: Gspan_format.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(EXECUTABLES) $(OBJS)
