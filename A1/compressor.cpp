#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <set> 
#include <algorithm>
#include "fptree_new.cpp"
#include <sstream>
#include <chrono>
using namespace std;

// map<vector<int>, int > compress_map;
map<vector<int>, int > compress_set;
uint64_t cnt = 0;
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
 
void compress_transactions(vector<vector<int> >& transactions , unordered_map<int,int> &freq , uint64_t numtransactions){
    
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t time_limit = 60*1000*30 ;
    // vector<float> support_values = {0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.01, 0.009, 0.007, 0.005, 0.003, 0.002, 0.001};
    int key = -1;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    vector<pair<vector<int> , int>>frequent_patterns ;
    uint64_t support = (int)(0.9*numtransactions);
    cout << "limiting support = " << (int)(0.001* numtransactions) <<endl;
    // vector<vector<int> > compressed_transactions = transactions;
    while(support >= (int)(0.001* numtransactions)){
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        cout<<"Support = "<<support<<"---------------"<<endl;
        if(elapsed_time.count() > time_limit){
            cout<<"Time limit exceeded for support = "<<endl;
            break;
        }
        // uint64_t support = (uint64_t)(support_values[isup]*numtransactions);
        fptree fpt;
        fpt.init(transactions, support , freq);
        frequent_patterns = fpt.pattern_mining(fpt, support, freq );

        cout<<"num patterns = "<<frequent_patterns.size()<<endl;
        std::cout << "Time taken for mining: " << elapsed_time.count() << " milliseconds" <<std::endl;
        uint64_t threshold = 0;
        int bound = 10000;
        for(int i=0; i<frequent_patterns.size(); i++){
            threshold += frequent_patterns[i].second; //*frequent_patterns[i].first.size();
        }

        if(threshold < bound){
            support = (int)(0.7*support);
            cout<<"skipping this support value due to very few patterns"<<endl;
            continue; //skip this support value
        }
        // sort frequent patterns according to size of frequent_patterns[i].first in decreasing order of size
        
        sort(frequent_patterns.begin(), frequent_patterns.end(), [](const pair<vector<int>, int> &left, const pair<vector<int>, int> &right) {
            return 0.1*left.first.size() + 0.9*left.second > 0.1*right.first.size() + 0.9*right.second;
        });
        vector<int> pattern;

        for(int ipattern=0; ipattern< min((int)frequent_patterns.size(), bound) ; ipattern++){
            pattern = frequent_patterns[ipattern].first;
            
            if(compress_set.find(pattern)==compress_set.end()){// new pattern
                decompress_map[key] = pattern;
                compress_set[pattern] = key;
                cnt += pattern.size() + 1;
                key--;
            }
            else{
                frequent_patterns.erase(frequent_patterns.begin()+ipattern);
                ipattern-=1;
            }
            
        }

        // vector<vector<int> > tmp_transactions;
        for(int itrans=0; itrans< transactions.size(); itrans++){
            // take the first 5000 patterns sorted by size of pattern
            for(int ipattern=0; ipattern< min((int)frequent_patterns.size(), bound) ; ipattern++){
                pattern = frequent_patterns[ipattern].first;
                if(isSubset(pattern, transactions[itrans] , freq)){
                    // if(compress_set.find(pattern)==compress_set.end()){// new pattern
                    //     decompress_map[key] = pattern;
                    //     compress_set[pattern] = key;
                    //     cnt += pattern.size() + 1;
                    //     transactions[itrans] =  substitute_key(pattern, transactions[itrans], key);
                    //     key--;
                    // }
                    // else
                    {
                        transactions[itrans] =  substitute_key(pattern, transactions[itrans], compress_set[pattern]);
                    }  
                }
            }
            // tmp_transactions.push_back(transactions[itrans]);
        }
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Time taken for compressing: " << elapsed_time.count() << " milliseconds" <<std::endl;
        support = (int)(0.8*support);
        // transactions = tmp_transactions;
    }
    ofstream outfile;
    outfile.open ("compressed_transactions.txt");
    if(outfile.is_open()){
        for(vector<int> trans: transactions){
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

