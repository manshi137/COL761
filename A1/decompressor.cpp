#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <set> 
#include <algorithm>
#include <sstream>
#include <chrono>
#include <functional>
using namespace std;

void decompress_main(string filepath , string output){
    ofstream outfile;
    outfile.open (output);

    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open the compressed file." << std::endl;
        // return 1;
    }

    map<int, vector<int> > decompress_map;
    int sizeofDM = 0;

    function<void(int)> decompress_help=[&](int elem){
        for(int elemmp: decompress_map[elem]){
            if(elemmp<0){
                decompress_help(elemmp);
            }
            else{
                outfile<<elemmp<<" ";
            }
        }
    };
    string line;
    int i = 0;
    while (getline(inputFile, line))
    {
        std::istringstream iss(line);
        std::vector<int> array;
        int element;

        if(i==0)
        {
            iss >> element;
            sizeofDM = element;
            i=1;
        }
        else if(i<=sizeofDM)
        {
            iss >> element;
            int key = element;
            while(iss >> element)
            {
                array.push_back(element);
            }
            decompress_map[key] = array;
            i++;

        }
        else
        {
        // Read elements from the current line
            while (iss >> element)
            {
                if(element<0){
                    decompress_help(element);
                }
                else{
                    outfile<<element<<" ";
                }
            }
            outfile<<'\n';
        }


    }

}


int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_argument>" << std::endl;
        return 1;
    }
    cout<<"Decompression starting.. \n";
    string compressed_file , output ; 
    compressed_file = argv[1];
    output = argv[2];
    decompress_main(compressed_file , output);
}