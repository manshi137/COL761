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
// map<int, vector<int> > decompress_map;
int compare (int a, int b , map<int , int >&freq)
{
    if(a == b )return 1 ;
    else if(freq[a] > freq[b] )return 0 ;
    else if(freq[a] == freq[b] && a < b) return 0; 
    else return -1;
}
bool isSubset(const std::set<int>& sub, const std::set<int>& sup , map<int , int> &freq ) {
    return includes(sup.begin() , sup.end() , sub.begin() , sub.end()); 
}
vector<int> substitute_key(set<int>& pattern, set<int>& trans, int& key){
    for(int elem: pattern){
        //if elem is not found in pattern, push it in ans vector
        trans.erase(elem);   
    }
    trans.insert(key);
    return vector<int>(trans.begin(),trans.end());
}
 
int compress_transactions(vector<vector<int> >& transactions , map<int,int> &freq , uint64_t numtransactions , string output){
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    uint64_t time_limit = 60*1000*45 ; //time limit
    int key = -1;
    uint64_t support = (int)(0.9*numtransactions);
    cout << "Limiting support = " << (int)(0.001* numtransactions) <<endl;

    while(support >= (uint64_t)(0.001* numtransactions))
    {
        
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        cout<<"Support = "<<support<<"=="<<support/numtransactions<<"---------------"<<endl;


        if(elapsed_time.count() > time_limit){
            cout<<"Time limit exceeded 1"<<endl;
            break;
        }
        fptree fpt;
        fpt.init(transactions, support , freq);
        vector<pair<vector<int> , int>>  frequent_patterns= fpt.pattern_mining(fpt, support, freq, end_time );

        cout<<"Number of frequent patterns mined = "<<frequent_patterns.size()<<endl;
        std::cout << "Time taken for mining: " << elapsed_time.count() << " milliseconds" <<std::endl;
        uint64_t threshold = 0;
        uint64_t bound = 10000;
        for(int i=0; i<frequent_patterns.size(); i++){
            threshold += frequent_patterns[i].second;
        }

        if(threshold < bound){
            support = (int)(0.6*support);
            cout<<"Skipping this support value due to very few patterns"<<endl;
            continue; 
        }
        
        sort(frequent_patterns.begin(), frequent_patterns.end(), [](const pair<vector<int>, int> &left, const pair<vector<int>, int> &right) {
            return left.second > right.second;
        });
        vector<int> pattern;
        for(int ipattern=0; ipattern< min((uint64_t)frequent_patterns.size(), bound) ; ipattern++)
        {
            pattern = frequent_patterns[ipattern].first;
            if(compress_set.find(pattern)==compress_set.end()){// new pattern
                compress_set[pattern] = key;
                cnt += pattern.size() + 1;
                key--;
            }
            else{
                cout << "errorfp ";
                frequent_patterns.erase(frequent_patterns.begin()+ipattern);
                ipattern-=1;
            }
            
        }

        for(int itrans=0; itrans< transactions.size(); itrans++)
        {
            end_time = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            if(elapsed_time.count() > time_limit){
                cout<<"Time limit exceeded 3"<<endl;
                break;
            }

            // take the first 5000 patterns sorted by size of pattern
            set<int>sup(transactions[itrans].begin(), transactions[itrans].end());
            for(auto ipattern : frequent_patterns){
                pattern = ipattern.first;
                set<int>sub(pattern.begin() , pattern.end()); 

                if(isSubset(sub, sup , freq)){
                    transactions[itrans] =  substitute_key(sub, sup, compress_set[pattern]);
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Time taken for compressing: " << elapsed_time.count() << " milliseconds" <<std::endl;
        support = (int)(0.7*support);
    }
    ofstream outfile;
    outfile.open (output);
    if(outfile.is_open()){
        outfile << compress_set.size()<< endl;
        for(auto c_set: compress_set)
        {
            outfile << c_set.second << " ";
            for(auto elem : c_set.first)
            {
                outfile << elem << " ";
            }
            outfile << "\n";
        }

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
    return cnt;
}