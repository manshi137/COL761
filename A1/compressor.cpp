#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <set> 
#include <algorithm>
#include "fptree_new.cpp"
#include <sstream>
using namespace std;

// map<vector<int>, int > compress_map;
set<vector<int> > compress_set;
map<int, vector<int> > decompress_map;
bool isSubset(const std::vector<int>& subset, const std::vector<int>& superset) {
    for (int element : subset) {
        if (std::find(superset.begin(), superset.end(), element) == superset.end()) {
            // Element from the subset not found in the superset
            return false;
        }
    }
    return true;
}
vector<int> substitute_key(vector<int>& pattern, vector<int>& trans, int& key){
    vector<int> ans;
    for(int elem: trans){
        //if elem is not found in pattern, push it in ans vector
        if(find(pattern.begin(), pattern.end(), elem) == pattern.end()){// elem not found in pattern
            ans.push_back(elem);
        }   
    }
    ans.push_back(key);
    return ans;
}
 
void compress_transactions(vector<vector<int> >& transactions , unordered_map<int,int> &freq , int numtransactions){
    vector<int> support_values = {(int)(0.8*numtransactions)};
    int key = -1;

    vector<vector<int> > compressed_transactions = transactions;
    for(int support: support_values){
        fptree fpt;
        fpt.init(compressed_transactions, support , freq);
        vector<vector<int> >frequent_patterns = fpt.pattern_mining(fpt, support, freq );


        vector<vector<int> > tmp_transactions;
        for(vector<int> trans: compressed_transactions){
            for(vector<int> pattern: frequent_patterns){
                if(pattern.size()>=2 && isSubset(pattern, trans)){
                    if(compress_set.find(pattern)==compress_set.end()){// new pattern
                        decompress_map[key] = pattern;
                        compress_set.insert(pattern);
                    }
                    trans =  substitute_key(pattern, trans, key);
                    key--;
                }
            }
            tmp_transactions.push_back(trans);
        }

        compressed_transactions = tmp_transactions;
    }
    ofstream outfile;
    outfile.open ("compressed_transactions.txt");
    if(outfile.is_open()){
        for(vector<int> trans: compressed_transactions){
            for(int elem: trans)outfile<<elem<<" ";
            outfile<<'\n';
        }
    }
    else{
        cout<<"Unable to open output file";
    
    }
}




void decompress_main(string filepath){
    ofstream outfile;
    outfile.open ("decompressed_transactions.txt");

    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open the compressed file." << std::endl;
        // return 1;
    }

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
    while (getline(inputFile, line))
    {
        std::istringstream iss(line);
        std::vector<int> array;

        // Read elements from the current line
        int element;
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

