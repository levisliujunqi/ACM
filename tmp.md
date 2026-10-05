## PAM

```c++
struct PAM{
    struct edge{
        int len,fail,cnt,sum;
        array<int, 26> nxt;
        edge(int len=0):len(len),fail(0),cnt(0),sum(0) {
            nxt.fill(0);
        }
    };
    vector<edge> node;
    string s;
    int last;
    PAM(){
        node.emplace_back();
        node.emplace_back(-1);//1:奇根
        node.emplace_back(0);//2:偶根
        node[1].fail=1;
        node[2].fail=1;
        last=2;
        s="#";
    }
    int getfail(int x,int pos){
        while(s[pos-node[x].len-1]!=s[pos]) {
            x=node[x].fail;
        }
        return x;
    }
    void add(char ch){
        s+=ch;
        int pos=s.size()-1;
        int c=ch-'a';
        int cur=getfail(last,pos);
        if (!node[cur].nxt[c]){
            int now=node.size();
            node.emplace_back(node[cur].len+2);
            if (node[now].len==1) {
                node[now].fail=2;
            }else{
                int x = getfail(node[cur].fail,pos);
                node[now].fail=node[x].nxt[c];
            }
            node[cur].nxt[c]=now;
        }
        last=node[cur].nxt[c];
        node[last].cnt++;
    }
};
```

## BurnSide 引理

对每种置换操作，统计操作前后不变的排列个数；把它们求和，再除以置换操作的总数，得到这些置换意义下本质不同的排列个数。
注意置换必须形成置换群。
## Polya 引理

对每种置换，若有 $c$ 个置换环、每个位置有 $k$ 种取值，那么该置换下不变方案数就是 $k^c$；把所有置换对应的 $k^c$ 求和，再除以置换总数，就是这些置换意义下本质不同的排列个数。
注意置换必须形成置换群。

## 全局最小割

```cpp
class Graph {
    vector<vector<int>> adj; // 邻接矩阵
    int n;
public:
    Graph(int n) : n(n), adj(n, vector<int>(n, 0)) {}
    void addEdge(int u, int v, int w) {
        adj[u][v] = adj[v][u] = w;
    }
    int stoerWagner() {
        int res = INT_MAX;
        for (int i = 0; i < n - 1; ++i) {
            vector<int> ma(n, 0);
            ma[0] = INT_MAX;  // 选择总是从 0 开始
            int s = -1, t = -1;
            for (int j = 0; j < n - i - 1; ++j) {
                int a = max_element(ma.begin(), ma.end()) - ma.begin();
                if (ma[a] == 0) return 0; // Graph is disconnected
                ma[a] = -1;
                if (j == n - i - 2) s = a;  // The second last node is s
                for (int k = 0; k < n; ++k) {
                    if (ma[k] >= 0) ma[k] += adj[a][k];
                }
            }
            t = max_element(ma.begin(), ma.end()) - ma.begin();
            res = min(res, ma[t]);
            // Merge nodes s and t
            for (int k = 0; k < n; ++k) {
                if (k != s && k != t) {
                    adj[s][k] += adj[t][k];
                    adj[k][s] = adj[s][k];
                    adj[t][k] = adj[k][t] = 0;
                }
            }
        } return res;
    }
};
```

