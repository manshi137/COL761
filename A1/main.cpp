#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
// #include "fptree_new.cpp"
#include "compressor.cpp"
using namespace std;
void c_p_c()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
}
int main()
{
    // c_p_c();
    std::vector<std::vector<int>> dataset;
    std::string line;
    std::ifstream inputFile("D_medium2.dat");
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return 1;
    }
    int ctint = 0 ; 
    auto start_time = std::chrono::high_resolution_clock::now();

    std::unordered_map<int , int > frequency ;
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
        
    fptree fpt;
    float support = 0.5 ; 
    uint64_t tot_sup = support * num_transaction ;
    cout<<"support , tot_transactions = "<<support<<" "<<num_transaction<<"\n";
    cout<<"support = "<<tot_sup <<'\n' ;
    start_time = std::chrono::high_resolution_clock::now();

    // fpt.init(dataset,(int)(support * num_transaction) , frequency );
    // fpt.create_conditionalFPT(1 , fpt  , tot_sup);
    // cout<<"mining started!"<<endl;
    // std::vector<std::vector<int>> p = fpt.pattern_mining(fpt , tot_sup ,frequency);
    // for(std::vector<int> v: p)
    // {
    //     for(int i: v)
    //     {
    //         cout<<i<<" ";
    //     }
    //     cout<<'\n';
    // }
    cout << "compressing file \n";

    compress_transactions(dataset , frequency, num_transaction);
    cout<<"initial num ints = "<<ctint<<endl;
    cout<<"decompressing file\n";
    decompress_main("compressed_transactions.txt");
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken for function: " << duration.count() << " milliseconds" << std::endl;
    
}