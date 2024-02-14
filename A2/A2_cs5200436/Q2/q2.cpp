#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

using namespace std;

int format_for_fsg(const std::string &input_file, const std::string &output_file)
{
    int out=0;
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);

    if (!infile.is_open() || !outfile.is_open())
    {
        std::cerr << "Error: Unable to open files." << std::endl;
        return 0;
    }

    std::string line;
    while (std::getline(infile, line))
    {
        if (line[0] == '#')
        {
            outfile << line << std::endl;
            outfile << "t" << std::endl;
            out+=1;
        }
        else if (line[0] == 'v')
        {
            outfile << line << std::endl;
        }
        else if (line[0] == 'e')
        {
            outfile << "u" << line.substr(1) << std::endl;
        }
    }

    infile.close();
    outfile.close();
    return out;
}

int read_tid(int out, string inf, string outf)
{
    ifstream infile(inf);   
    ofstream outfile(outf); 
    if (!infile.is_open())
    {
        cerr << "Error opening input file!" << endl;
        return 1;
    }

    map<int, set<string>> graphToSubgraphs;
    vector<string> subgraphs;
    int i = 0;
    while (i<out)
    {
        graphToSubgraphs[i] = {};
        i += 1;
    }

    if (!outfile.is_open())
    {
        cerr << "Error creating output file!" << endl;
        return 1;
    }

    string line;

    int b=0;
    while (getline(infile, line))
    {
        b++;
        if(b>100)
        {
            break;
        }

        stringstream ss(line);
        string subgraphId;
        ss >> subgraphId;
        subgraphs.push_back(subgraphId);

        int graphId;

        while (ss >> graphId)
        {
            graphToSubgraphs[graphId].insert(subgraphId);
        }
    }

    for (auto entry : graphToSubgraphs)
    {
        
        outfile << entry.first << " # ";
        for (string subgraphId : subgraphs)
        {
            if (entry.second.count(subgraphId) == 0)
            {
                outfile << 0 << " ";
            }
            else
            {
                outfile << 1 << " ";
            }
        }
        outfile << endl;
    }

    infile.close();
    outfile.close();

    return 0;
}

void run_command(const std::string &input_file, const int &threshold)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string command = "./Q2/fsg -s " + std::to_string(threshold) + " -t -x " + input_file;
    int ret_code = std::system(command.c_str());
    

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    if (ret_code == 0)
    {
        std::cout << "Threshold: " << threshold << "% - Runtime: " << duration << " seconds" << std::endl;
    }
    else
    {
        std::cerr << "Error running command for threshold " << threshold << "%: Returned code " << ret_code << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <output_file>" << std::endl;
        return 1;
    }

    int out = format_for_fsg(argv[1], "temp.txt");

    int threshold = 20;
            run_command("temp.txt", threshold);
            read_tid(out, "temp.tid", argv[2]);

    return 0;
}
