#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;

int main(int argc, char *argv[])
{
    map<string  , int> vertices ;
    int cur_node = 0 ;
     if (argc != 2) {
        cerr << "Usage: " << argv[0] << " input_file output_file" << endl;
        return 1;
    }
    ifstream infile(argv[1]);
    ofstream outfile_FSG("Yeast_FSG.txt");

    std::string line;
    int count = 0;
    int v = 0;
    int e = 0;
    int i = 0;

    if (!infile.is_open() || !outfile_FSG.is_open()) {
        std::cerr << "Error: Unable to open files." << std::endl;
        return 0;
    }

    while (std::getline(infile, line)) {
        if (line[0] == '#') {
            outfile_FSG << line << std::endl;
            outfile_FSG << "t\n";
            count++;
        } else if (count == 1) {
            count++;
            v = std::stoi(line);
        } else if (count == 2) {
            if (i == v) {
                count++;
                i = 0;
                e = std::stoi(line);
                continue;
            }
            if(vertices.find(line ) == vertices.end())
            {
                vertices[line] = cur_node;
                cur_node++; 
            }
            outfile_FSG << "v " << i << " " << vertices[line] << std::endl;
            i++;
        } else if (count == 3) {
            if (i == e) {
                i = 0;
                count = 0;
                continue;
            }
            outfile_FSG << "u " << line << std::endl;
            i++;
        }
    }
    return 0;
}