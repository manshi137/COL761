#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "fptree.cpp"
using namespace std;
void c_p_c()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
}
int main()
{
    c_p_c();
    std::vector<std::vector<int>> dataset;
    std::string line;
    std::ifstream inputFile("D_small.dat");
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return 1;
    }
    while (std::getline(inputFile, line))
    {
        std::istringstream iss(line);
        std::vector<int> array;

        // Read elements from the current line
        int element;
        while (iss >> element)
        {
            array.push_back(element);
        }

        // Add the array to the dataset
        dataset.push_back(array);
    }
    // for (vector<int> v : dataset)
    // {
    //     for (int i : v)
    //         cout << i << " ";
    //     cout << '\n';
    // }
    fptree fpt;
    fpt.init(dataset,200);
    // cout<<
    // for(auto s :fpt.table)
    // {
    //     cout<<s.first<<" "<<s.second.first->id <<'\n';
    // }
    // fpt.dfs(fpt.root);
    // fpt.create_conditionalFPT(3,fpt);

    vector<vector<int>> patterns = fpt.pattern_mining(fpt,200);
    // for(int i=0;i<patterns.size();i++)
    // {
    //     for(int j=0;j<patterns[i].size();j++)
    //     {
    //         cout << patterns[i][j] << " ";
    //     }
    //     cout << "\n";
    // }
}
