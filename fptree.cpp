#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <set> 
#include <algorithm>
using namespace std;

struct node{
    int id; //item number
    int freq; //frequency of the item
    node* parent; 
    node* next_node; //node of the same item.....linked list
    map<int,node*> child; //here int id the id of the child and corresponding point is node*

    node(int id, int freq, node* parent) : id(id), freq(freq), parent(parent), next_node(nullptr)  {}

};

struct fptree{
    node* root;
    map<int,pair<node*,node*> > table; //this table will store the first and current last of particular item
    map<int,int> freq_table; 

    fptree() : root(new node(-1, 0, nullptr)) {}  // Assuming -1 is a special value for the root node

    void init(vector<pair<vector<int>,int>> &transactions,map<int,int> &freq_t,int support)
    {
        for(pair<vector<int>,int> trs : transactions)
        {
            node* curr_node = root ; 
            for(int object : trs.first)
            {
                if(freq_t.find(object) == freq_t.end() || freq_t[object] >= support)
                {
                    if(curr_node->id == object)
                    {
                    }
                    else if(curr_node->child.find(object) == curr_node->child.end()) // if object is not present in the child of curr_node
                    {
                        curr_node->child[object] = new node(object,trs.second,curr_node);
                        if(table.find(object) == table.end())
                        {
                            table[object].first = curr_node->child[object];
                            table[object].second = curr_node->child[object];
                        }
                        else
                        {
                            table[object].second->next_node = curr_node->child[object];
                            table[object].second = curr_node->child[object];

                        }
                        curr_node = curr_node->child[object];
                    }
                    else
                    {
                        curr_node->child[object]->freq+=trs.second;
                        curr_node = curr_node->child[object];
                    }
                }
                else
                {
                    freq_t.erase(object);
                }
            }
            freq_table = freq_t;
        }        
    }

    void init(vector<vector<int>> &transaction, int support)
    {
        vector<vector<int>> trans_freq_adj;
        map<int, int> freq;
        for (vector<int> trs : transaction)
        {
            for (int obj : trs)
            {
                freq[obj]++;
            }
        }
        for (vector<int> trs : transaction)
        {
            vector<int> temp_trans;
            for (int obj : trs)
            {
                if (freq[obj] < support)
                    continue;
                temp_trans.push_back(obj);
            }
            trans_freq_adj.push_back(temp_trans);
        }
        // sort transaction with freq
        for (vector<int> &trs : trans_freq_adj)
        {
            sort(trs.begin(), trs.end(), [&freq](int a, int b)
                 { return freq[a] > freq[b]; });
        }
        for (vector<int> trs : trans_freq_adj)
        {
            node *curr_node = root;
            for (int object : trs)
            {
                if (curr_node->id == object)
                {
                }
                else if (curr_node->child.find(object) == curr_node->child.end()) // if object is not present in the child of curr_node
                {
                    curr_node->child[object] = new node(object, 1, curr_node);
                    if (table.find(object) == table.end())
                    {
                        table[object].first = curr_node->child[object];
                        table[object].second = curr_node->child[object];
                        freq_table[object] = 1;
                    }
                    else
                    {
                        table[object].second->next_node = curr_node->child[object];
                        table[object].second = curr_node->child[object];
                        freq_table[object] += 1;
                    }
                    curr_node = curr_node->child[object];
                }
                else
                {
                    curr_node->child[object]->freq += 1;
                    freq_table[object] += 1;

                    curr_node = curr_node->child[object];
                }
            }
        }
    }

    void dfs(node* curr_node) {
        cout << curr_node->id << ":" << curr_node->freq << " ";
        for (auto it = curr_node->child.begin(); it != curr_node->child.end(); ++it) {
            dfs(it->second);
        }
    }

    fptree create_conditionalFPT(int id, fptree FPT,int support)
    {
        // making list of conditional branches along with frequency
        fptree cond_fpt;
        // map<int,pair<node*,node*> > cond_table;
        map<int,int> freq_t;
        node* last_node = FPT.table[id].first;
        vector<pair<vector<int>,int>> cond_branches;
        while(last_node!=nullptr)
        {
            int freq = last_node->freq;
            vector<int> cond_br;
            node* curr_node = last_node->parent;
            while(curr_node->id != -1)
            {
                cond_br.push_back(curr_node->id);
                if(freq_t.find(curr_node->id) == freq_t.end())
                {
                    freq_t[curr_node->id] = freq;
                }
                else
                {
                    freq_t[curr_node->id] += freq;
                }
                // cout << curr_node->id;
                // int freq = min(freq,curr_node->freq);
                curr_node = curr_node->parent;
            }
            reverse(cond_br.begin(),cond_br.end());
            last_node = last_node->next_node;
            cond_branches.push_back({cond_br,freq});
        }

        cond_fpt.init(cond_branches,freq_t,support);
        // cond_fpt.dfs(cond_fpt.root);  // to print      
        return cond_fpt;
    }

    vector<vector<int>> pattern_mining(fptree FPT, int support)
    {
        vector<vector<int>> out;
       
        for(auto object : FPT.table)
        {
            if(FPT.freq_table[object.first] >= support)
            {
                out.push_back({object.first});
                fptree conditional_fpt = create_conditionalFPT(object.first,FPT, support);
                vector<vector<int>> temp_out = pattern_mining(conditional_fpt,support);


                for(vector<int> &v : temp_out)
                {
                    if(v.size()==0) continue;
                    v.push_back(object.first);
                    out.push_back(v);
                }
            }
        }
        return out;
    }

};




