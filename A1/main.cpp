#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
// #include "fptree_new.cpp"
#include "compressor.cpp"
// #include "decompressor.cpp"
using namespace std;
void c_p_c()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
}
int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_argument>" << std::endl;
        return 1;
    }

    // c_p_c();
    std::vector<std::vector<int>> dataset;
    std::string line;
    std::ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open the input file." << std::endl;
        return 1;
    }
    int ctint = 0 ; 
    auto start_time = std::chrono::high_resolution_clock::now();

    std::map<int , int > frequency ;
    uint64_t num_transaction = 0 ; 
    while (std::getline(inputFile, line))
    {
        std::istringstream iss(line);
        std::vector<int> array;

        // Read elements from the current line
        int element;
        while (iss >> element)
        {
            array.push_back(element);
            frequency[element]++; 
            ctint++;
        }
        num_transaction++;  

        // Add the array to the dataset
        dataset.push_back(array);
    }
    inputFile.close();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken input: " << duration.count() << " milliseconds" << std::endl;
        
    start_time = std::chrono::high_resolution_clock::now();
    cout << "Compression starting... \n";

    int cnt = compress_transactions(dataset , frequency, num_transaction, argv[2]);
    cout<<"Initial number of ints = "<<ctint<<endl;
    cout << "Compression  = " << 1 - (float)cnt/(float)ctint << endl;
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken for function: " << duration.count() << " milliseconds" << std::endl;
    
}