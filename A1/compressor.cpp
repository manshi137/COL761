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
map<vector<int>, int > compress_set;
int cnt = 0;
map<int, vector<int> > decompress_map;
int compare (int a, int b , unordered_map<int , int >&freq)
{
    if(a == b )return 1 ;
    else if(freq[a] > freq[b] )return 0 ;
    else if(freq[a] == freq[b] && a < b) return 0; 
    else return -1;
}
bool isSubset(const std::vector<int>& subset, const std::vector<int>& superset , unordered_map<int , int> &freq ) {
    // int i = 0 ;
    // int j= 0 ;
    // while(i < superset.size() && j < subset.size())
    // {

    //     if(compare(superset[i] ,  subset[j] , freq) == 1) // both equa
    //     {
    //         i++;
    //         j++;
    //     }
    //     else if(compare(superset[i] ,  subset[j] , freq) == 0)
    //     {
    //         i++;
    //     }
    //     else 
    //     {
    //         return false;
    //     }
    // }
    // if(j == subset.size())return true; 
    // else return false ;
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
    vector<int> support_values = {(int)(0.05*numtransactions)};
    int key = -1;

    vector<vector<int> > compressed_transactions = transactions;
    for(int support: support_values){
        fptree fpt;
        fpt.init(compressed_transactions, support , freq);
        vector<pair<vector<int> , int>>frequent_patterns = fpt.pattern_mining(fpt, support, freq );


        vector<vector<int> > tmp_transactions;
        for(vector<int> trans: compressed_transactions){
            for(pair<vector<int>,int> pat: frequent_patterns){
                vector<int> pattern = pat.first;
                if(pattern.size()>=2 && isSubset(pattern, trans , freq)){
                    if(compress_set.find(pattern)==compress_set.end()){// new pattern
                        decompress_map[key] = pattern;
                        compress_set[pattern] = key;
                        cnt += pattern.size() + 1;
                    trans =  substitute_key(pattern, trans, key);
                        key--;
                    }
                    else{
                    trans =  substitute_key(pattern, trans, compress_set[pattern]);
                        
                    }
                    
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
            for(int elem: trans)
            {
                outfile<<elem<<" ";
                cnt+=1;
            }
            outfile<<'\n';
        }
    }
    else{
        cout<<"Unable to open output file";
    
    }
    cout << "cnt after cmpression = "<< cnt << endl;
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

