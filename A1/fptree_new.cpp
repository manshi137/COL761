#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <chrono>
#include <memory>
using namespace std;


typedef std::vector<std::vector<int>> Transactions;
typedef std::vector<std::pair<std::vector<int>, int>> TransactionsPair;
struct node
{
    int id;   // item number
    int freq; // frequency of the item
    std::shared_ptr<node> parent;

    std::shared_ptr<node> next_node;                      // node of the same item.....linked list
    std::unordered_map<int, std::shared_ptr<node>> child; // here int id the id of the child and corresponding point is node*

    node(int id, int freq, std::shared_ptr<node> parent) : id(id), freq(freq), parent(parent), next_node(nullptr) {}
};

typedef std::unordered_map<int, std::pair<std::shared_ptr<node>, std::shared_ptr<node>>> NodeTable;
struct fptree
{
    std::shared_ptr<node> root;
    NodeTable table; // this table will store the first and current last of particular item
    std::unordered_map<int, int> freq_table;

    fptree() : root(new node(-1, 0, nullptr)) {} // Assuming -1 is a special value for the root node
    bool single_p(const std::shared_ptr<node> &node)
    {
        if (node->child.size() == 0)
            return true;
        if (node->child.size() > 1)
            return false;
        return single_p((*(node->child.begin())).second);
    }
    bool single_p(fptree &fptree)
    {
        return root->child.empty() || single_p(fptree.root);
    }
    void init(Transactions &transactions, int support, std::unordered_map<int, int> &freq)
    {
        for (std::vector<int> &trs : transactions)
        {
            // std::cout<<"Newtrans"<<std::endl;
            std::vector<int> tmp_trs;
            for (int obj : trs)
            {
                if (freq[obj] >= support)
                {
                    tmp_trs.push_back(obj);
                }
            }
            sort(tmp_trs.begin(), tmp_trs.end(), [&freq](int a, int b)
                 { if(freq[a] == freq[b] ) return a<b; return freq[a] > freq[b]; });
            auto curr_node = root;
            for (int object : tmp_trs)
            {
                if (curr_node->child.find(object) == curr_node->child.end())
                {
                    const auto new_child_node = std::make_shared<node>(object, 1, curr_node);
                    curr_node->child[object] = new_child_node;

                    if (table.count(object))
                    {

                        table[object].second->next_node = curr_node->child[object];
                        table[object].second = curr_node->child[object];
                        freq_table[object] += 1;
                    }
                    else
                    {
                        table[object].first = curr_node->child[object];
                        table[object].second = curr_node->child[object];
                        freq_table[object] = 1;
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

    void init(TransactionsPair &transactions, int support, std::unordered_map<int, int> freq)
    {
        for (std::pair<std::vector<int>, int> &trs : transactions)
        {
            std::vector<int> tmp_trs;
            for (int obj : trs.first)
            {
                if (freq[obj] >= support)
                {
                    tmp_trs.push_back(obj);
                }
            }
            int freq_obj = trs.second;
            sort(tmp_trs.begin(), tmp_trs.end(), [&freq](int a, int b)
                 {if(freq[a] == freq[b] ) return a<b; return freq[a] > freq[b]; });
            auto curr_node = root;
            for (int object : tmp_trs)
            {
                if (curr_node->child.find(object) == curr_node->child.end())
                {
                    const auto new_child_node = std::make_shared<node>(object, freq_obj, curr_node);
                    curr_node->child[object] = new_child_node;

                    if (table.count(object))
                    {

                        table[object].second->next_node = curr_node->child[object];
                        table[object].second = curr_node->child[object];
                        freq_table[object] += freq_obj;
                    }
                    else
                    {
                        table[object].first = curr_node->child[object];
                        table[object].second = curr_node->child[object];
                        freq_table[object] = freq_obj;
                    }
                    curr_node = curr_node->child[object];
                }

                else
                {
                    curr_node->child[object]->freq += freq_obj;
                    freq_table[object] += freq_obj;

                    curr_node = curr_node->child[object];
                }
            }
        }
    }

    fptree create_conditionalFPT(int object, fptree fpt, int support)
    {

        std::shared_ptr<node> last_node = fpt.table[object].first;
        std::unordered_map<int, int> freq_tab;
        TransactionsPair cond_branches;
        while (last_node != nullptr)
        {
            int freq = last_node->freq;
            std::vector<int> cond_br;
            std::shared_ptr<node> curr_node = last_node->parent;
            while (curr_node->id != -1)
            {
                cond_br.push_back(curr_node->id);
                if (freq_tab.find(curr_node->id) == freq_tab.end())
                {
                    freq_tab[curr_node->id] = freq;
                }
                else
                {
                    freq_tab[curr_node->id] += freq;
                }
                curr_node = curr_node->parent;
            }
            reverse(cond_br.begin(), cond_br.end());
            last_node = last_node->next_node;
            cond_branches.push_back({cond_br, freq});
        }
        fptree cond_fpt;
        cond_fpt.init(cond_branches, support, freq_tab);
        return cond_fpt;
    }

    std::vector<pair<std::vector<int>, int>> pattern_mining(fptree FPT, int support, std::unordered_map<int, int> &freq, int n = 1)
    {
        std::vector<pair<std::vector<int>, int>> out;

        if (single_p(FPT))
        {

            // auto node = (*(fptree.root->child.begin()));
            // while(node)
            // {
            //     int object = node->id;
            //     int freq = node->freq ;
            //     int cur_
            // }
            return {};
        }
        else
        {

            for (auto object : FPT.table)
            {
                if (FPT.freq_table[object.first] >= support)
                {

                    if(n!=1) out.push_back({{object.first} , FPT.freq_table[object.first]});
                    fptree conditional_fpt = create_conditionalFPT(object.first, FPT, support);
                    std::vector<pair<std::vector<int>, int>> temp_out = pattern_mining(conditional_fpt, support, freq, 0);
                    for (auto &v : temp_out)
                    {
                        if(n==1) 
                        {
                            v.second *= (v.first.size() + 1);
                            // auto it = std::find_if(out.begin(), out.end(), v);
                            auto it = std::find_if(out.begin(), out.end(), [&v](const auto& element) {
                                return v.first == element.first;
                            });
                            if(it != out.end())
                            {
                                if(v.second = it->second)
                                {
                                    it->first = v.first;
                                }
                                else if(it->second - v.second < support)
                                {
                                    *it = v;
                                }
                                else
                                {
                                    it->second = it->second - v.second;
                                }
                                continue;
                            }
                        }
                        v.first.push_back(object.first);
                        std::sort(v.first.begin(), v.first.end(), [&freq](int a, int b)
                                  { if(freq[a] == freq[b] ) return a<b; return freq[a] > freq[b]; });

                        out.push_back(v);
                    }
                }
            }
            return out;
        }
    }
};