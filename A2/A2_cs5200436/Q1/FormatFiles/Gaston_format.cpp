#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;
map<string, int> mp;
long long c=0;

int main(int argc, char *argv[])
{
     if (argc != 2) {
        cerr << "Usage: " << argv[0] << " input_file output_file" << endl;
        return 1;
    }
    ifstream infile(argv[1]);
    ofstream outfile("Yeast_Gaston.txt");

    std::string line;
    std::string transaction_line = "Transaction t The beginning of a transaction\n";

    int count = 0;
    int v = 0;
    int e = 0;
    int i = 0;
    int numt=0;
    while (std::getline(infile, line)) {
        if (line[0] == '#') {
            // outfile << line << std::endl;
            outfile << "t # "<<numt++<<std::endl;
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
            if(mp.find(line) == mp.end())
            {
                mp[line]=c;
                c++;
            }
            // cout<<
            outfile << "v " << i << " " << to_string(mp[line]) << std::endl;
            i++;
        } else if (count == 3) {
            if (i == e) {
                i = 0;
                count = 0;
                continue;
            }
            outfile << "e " << line << std::endl;
            i++;
        }
    }
    return 0;
}