# 数据结构

## 树状数组

区间和

```cpp
struct BIT{
    vector<int> tree;
    int n;
    inline int lowbit(int x){
        return x&(-x);
    }
    BIT(int n){
        this->n=n;
        tree.resize(n+1,0);
    }
    void update(int x,int delta){
        for(int i=x;i<=n;i+=lowbit(i)) tree[i]+=delta; 
    }
    int query(int x,int y){
        if(x>y) return 0;
        int ans=0;
        for(int i=y;i>=1;i-=lowbit(i)) ans+=tree[i];
        for(int i=x-1;i>=1;i-=lowbit(i)) ans-=tree[i];
        return ans;
    }
};
```

前缀区间最值
修改只能往大了修改，$newvalue>=oldvalue$

```c++
struct BIT{
    vector<int> tree;
    int n;
    inline int lowbit(int x){
        return x&(-x);
    }
    BIT(int n){
        this->n=n+1;
        tree.resize(n+1,-INF);
    }
    void update(int x,int y){
        ++x;
        tree[x]=y;
        for(int i=x;i<=n;i+=lowbit(i)){
            tree[i]=max(tree[i],y);
        }
    }
    void clear(int x){
        ++x;
        tree[x]=-INF;
        for(int i=x;i<=n;i+=lowbit(i)){
            tree[i]=-INF;
        }
    }
    int query(int x){
        ++x;
        if(x<1) return -INF;
        int ans=-INF;
        for(int i=x;i>=1;i-=lowbit(i)){
            ans=max(ans,tree[i]);
        }
        return ans;
    }
};
```

## 线段树

```cpp
struct SegmentTree {
    int lp(int x) {
        return x << 1;
    }
    int rp(int x) {
        return x << 1 | 1;
    }
    struct edge {
        int sum;
        edge() {
            sum = 0;
        }
    };
    vector<int> lazy;
    vector<edge> node;
    int n;
    void pushup(int id, int l, int r) {
        node[id].sum = node[lp(id)].sum + node[rp(id)].sum;
    }
    void pushdown(int id, int l, int r) {
        if (lazy[id]) {
            int mid = l + (r - l >> 1);
            lazy[lp(id)] += lazy[id];
            lazy[rp(id)] += lazy[id];
            node[lp(id)].sum += (mid - l + 1) * lazy[id];
            node[rp(id)].sum += (r - mid) * lazy[id];
            lazy[id] = 0;
        }
    }
    SegmentTree(int n) : n(n) {
        node.resize((n << 2) + 5);
        lazy.assign((n << 2) + 5, 0);
    }
    SegmentTree() {}
    void init(vector<int> &v) {
        auto buildtree = [&](auto self, int id, int l, int r) -> void {
            lazy[id] = 0;
            if (l == r) {
                node[id].sum = v[l];
                return;
            }
            int mid = l + (r - l >> 1);
            self(self, lp(id), l, mid);
            self(self, rp(id), mid + 1, r);
            pushup(id, l, r);
        };
        buildtree(buildtree, 1, 1, n);
    }
    SegmentTree(int n, vector<int> &v) : n(n) {
        node.resize((n << 2) + 5);
        lazy.assign((n << 2) + 5, 0);
        init(v);
    }
    void update(int id, int l, int r, int x, int y, int delta) {
        if (x > y)
            return;
        if (x <= l && r <= y) {
            lazy[id] += delta;
            node[id].sum += delta * (r - l + 1);
            return;
        }
        pushdown(id, l, r);
        int mid = l + (r - l >> 1);
        if (x <= mid)
            update(lp(id), l, mid, x, y, delta);
        if (y > mid)
            update(rp(id), mid + 1, r, x, y, delta);
        pushup(id, l, r);
    }
    int query(int id, int l, int r, int x, int y) {
        if (x > y)
            return 0;
        if (x <= l && r <= y)
            return node[id].sum;
        pushdown(id, l, r);
        int mid = l + (r - l >> 1);
        int ans = 0;
        if (x <= mid)
            ans += query(lp(id), l, mid, x, y);
        if (y > mid)
            ans += query(rp(id), mid + 1, r, x, y);
        return ans;
    }
    // 第一个满足前缀和>x的位置,找不到n+1
    int upper_bound(int id, int l, int r, int x) {
        if (l == r) {
            if (node[id].sum > x)
                return l;
            else
                return n + 1;
        }
        pushdown(id, l, r);
        int mid = l + (r - l >> 1);
        if (node[lp(id)].sum > x) {
            return upper_bound(lp(id), l, mid, x);
        } else {
            return upper_bound(rp(id), mid + 1, r, x - node[lp(id)].sum);
        }
    }
};
```

## 线段树标记永久化

```cpp
struct SegmentTree{
    struct edge{
        int sum,lazy;
        edge(){
            sum=lazy=0;
        }
    };
    int n;
    vector<edge> node;
    SegmentTree(int n):n(n){
        node.resize(n*4+5);
    }
    void update(int x,int y,int delta){
        auto upd=[&](auto self,int id,int l,int r,int x,int y,int delta){
            node[id].sum+=delta*(min(y,r)-max(l,x)+1);
            if(x<=l&&r<=y){
                node[id].lazy+=delta;
                return;
            }
            int mid=l+(r-l>>1);
            if(x<=mid) self(self,id<<1,l,mid,x,y,delta);
            if(y>mid) self(self,id<<1|1,mid+1,r,x,y,delta);
        };
        upd(upd,1,1,n,x,y,delta);
    }
    int query(int x,int y){
        auto que=[&](auto self,int id,int l,int r,int x,int y,int lz){
            if(x<=l&&r<=y){
                return node[id].sum+lz*(r-l+1);
            }
            int ans=0;
            int mid=l+(r-l>>1);
            lz+=node[id].lazy;
            if(x<=mid){
                ans+=self(self,id<<1,l,mid,x,y,lz);
            }
            if(y>mid){
                ans+=self(self,id<<1|1,mid+1,r,x,y,lz);
            }
            return ans;
        };
        return que(que,1,1,n,x,y,0);
    }
};
```

## 线段树合并

update和merge之后一定要记得接收新的根编号。

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct SegmentTree{
    struct edge{
        int maxn,maxnpos,lson,rson;
        edge():maxn(0),maxnpos(0),lson(0),rson(0){}
    };
    int n;
    vector<edge> node;
    SegmentTree(int n):n(n),node(1){}
    void pushup(int id,int l,int r){
        node[id].maxn=0;
        node[id].maxnpos=0;
        if(node[id].lson){
            if(node[id].maxn<node[node[id].lson].maxn){
                node[id].maxn=node[node[id].lson].maxn;
                node[id].maxnpos=node[node[id].lson].maxnpos;
            }
        }
        if(node[id].rson){
            if(node[id].maxn<node[node[id].rson].maxn){
                node[id].maxn=node[node[id].rson].maxn;
                node[id].maxnpos=node[node[id].rson].maxnpos;
            }
        }
    }
    int update(int id,int l,int r,int x,int delta){
        if(id==0){
            id=node.size();
            node.emplace_back();
        }
        if(l==r){
            node[id].maxn+=delta;
            node[id].maxnpos=l;
            return id;
        }
        int mid=l+(r-l>>1);
        if(x<=mid) node[id].lson=update(node[id].lson,l,mid,x,delta);
        else node[id].rson=update(node[id].rson,mid+1,r,x,delta);
        pushup(id,l,r);
        return id;
    }
    int merge(int a,int b,int l,int r){
        if(!a&&!b) return 0;
        if(!a) return b;
        if(!b) return a;
        if(l==r){
            node[a].maxn+=node[b].maxn;
            return a;
        }
        int mid=l+(r-l>>1);
        node[a].lson=merge(node[a].lson,node[b].lson,l,mid);
        node[a].rson=merge(node[a].rson,node[b].rson,mid+1,r);
        pushup(a,l,r);
        return a;
    }
    int query(int id){
        return node[id].maxnpos;
    }
};
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> v(n+1);
    vector<int> dep(n+1);
    vector<vector<int>> fa(n+1,vector<int>(25));
    vector<map<int,int>> mp(n+1);
    for(int i=1;i<n;i++){
        int x,y;
        cin>>x>>y;
        v[x].push_back(y);
        v[y].push_back(x);
    }
    function<void(int,int)> dfs=[&](int x,int father){
        dep[x]=dep[father]+1;
        fa[x][0]=father;
        for(int i=1;i<=20;i++){
            fa[x][i]=fa[fa[x][i-1]][i-1];
        }
        for(int &to:v[x]){
            if(to==father) continue;
            dfs(to,x);
        }
    };
    dfs(1,0);
    auto LCA=[&](int x,int y){
        if(dep[x]<dep[y]) swap(x,y);
        for(int i=20;i>=0;i--){
            if(dep[fa[x][i]]>=dep[y]){
                x=fa[x][i];
            }
        }
        if(x==y) return x;
        for(int i=20;i>=0;i--){
            if(fa[x][i]!=fa[y][i]){
                x=fa[x][i];
                y=fa[y][i];
            }
        }
        return fa[x][0];
    };
    for(int i=1;i<=m;i++){
        int x,y,z;
        cin>>x>>y>>z;
        mp[x][z]++;
        mp[y][z]++;
        int l=LCA(x,y);
        mp[l][z]--;
        mp[fa[l][0]][z]--;
    }
    vector<int> head(n+1),ans(n+1);
    SegmentTree tree(1e5);
    function<void(int,int)> dfs1=[&](int x,int father){
        for(auto &to:v[x]){
            if(to==father) continue;
            dfs1(to,x);
            head[x]=tree.merge(head[x],head[to],1,1e5);
        }
        for(auto &[p,q]:mp[x]){
            head[x]=tree.update(head[x],1,1e5,p,q);
        }
        ans[x]=tree.query(head[x]);
    };
    dfs1(1,0);
    for(int i=1;i<=n;i++) cout<<ans[i]<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 并查集

```cpp
//带权并查集
//sz表示祖先节点个数
struct DSU{
    vector<int> fa;
    vector<int> sz;
    int n;
    DSU(int n){
        this->n=n;
        fa.resize(n+1);
        sz.resize(n+1,0);
        for(int i=0;i<=n;i++) fa[i]=i;
    }
    int find(int x){
        if(fa[x]==x) return x;
        int fax=fa[x];
        fa[x]=find(fa[x]);
        sz[x]+=sz[fax];
        return fa[x];
    }
    bool merge(int x,int y){
        int fax=find(x);
        int fay=find(y);
        if(fax==fay) return 0;
        fa[fax]=fay;
        sz[fax]=1;
        return 1;
    }
};
```

```cpp
//按秩合并+路径压缩，rank表示子树深度
struct DSU{
    vector<int> fa;
    vector<int> rank;
    int n;
    DSU(int n){
        this->n=n;
        fa=vector<int>(n+1);
        rank=vector<int>(n+1,0);
        for(int i=0;i<=n;i++) fa[i]=i;
    }
    int find(int x){
        return fa[x]==x?x:fa[x]=find(fa[x]);
    }
    bool merge(int x,int y){
        int fax=find(x);
        int fay=find(y);
        if(fax==fay) return 0;
        if(rank[fax]<rank[fay]) fa[fax]=fay;
        else{
            fa[fay]=fax;
            if(rank[fax]==rank[fay]) rank[fax]++;
        }
        return 1;
    }
};
```

```cpp
//可撤销并查集，启发式合并，find是logn的
struct DSU{
    vector<int> fa;
    vector<int> sz;
    int n;
    stack<array<int,4>> st;
    DSU(int n){
        this->n=n;
        fa=vector<int>(n+1);
        sz=vector<int>(n+1,1);
        for(int i=0;i<=n;i++) fa[i]=i;
    }
    int find(int x){
        if(fa[x]==x) return x;
        else return find(fa[x]);
    }
    bool merge(int x,int y){
        int fax=find(x);
        int fay=find(y);
        if(fax==fay){
            st.push({-1,-1,-1,-1});
            return 0;
        }
        if(sz[fax]>sz[fay]) swap(fax,fay);
        st.push({fax,fa[fax],fay,sz[fax]});
        fa[fax]=fay;
        sz[fay]+=sz[fax];
        return 1;
    }
    void pop(){
        auto [a,b,c,d]=st.top();
        st.pop();
        if(a==-1) return;
        fa[a]=b;
        sz[c]-=d;
    }
};
```

## 主席树

```cpp
// 要注意当k>区间元素个数的时候kth会返回值域上界/下界
struct PresidentTree{
    vector<int> node;
    vector<int> lson, rson;
    vector<int> head;
    int L, R;
    void pushup(int id, int l, int r){
        node[id] = 0;
        if (lson[id])
            node[id] += node[lson[id]];
        if (rson[id])
            node[id] += node[rson[id]];
    }
    PresidentTree(int l, int r) : L(l), R(r){
        node.resize(2);
        lson.resize(2);
        rson.resize(2);
        head.resize(1);
        node[1] = 0;
        head[0] = 1;
    }
    // 在baseid版本上更新，x这个位置+=y
    void update(int baseid, int x, int y){
        auto updatenode = [&](auto &&self, int base, int l, int r, int x, int y){
            int now = node.size();
            node.emplace_back();
            lson.emplace_back();
            rson.emplace_back();
            if (l == r){
                if (!base)
                    node[now] = y;
                else
                    node[now] = node[base] + y;
                return now;
            }
            int mid = l + (r - l >> 1);
            if (x <= mid){
                lson[now] = self(self, lson[base], l, mid, x, y);
                rson[now] = rson[base];
            }
            else{
                lson[now] = lson[base];
                rson[now] = self(self, rson[base], mid + 1, r, x, y);
            }
            pushup(now, l, r);
            return now;
        };
        head.push_back(updatenode(updatenode, head[baseid], L, R, x, y));
    }
    // 加一个版本
    void push_back(int x, int y) {
        update((int)head.size() - 1, x, y);
    }
    // 询问id版本值域[x,y]的权值
    int query(int id, int x, int y){
        if(x > y) return 0;
        auto querynode = [&](auto &&self, int base, int l, int r, int x, int y){
            if (x <= l && r <= y)
                return node[base];
            int mid = l + (r - l >> 1);
            int ans = 0;
            if (x <= mid && lson[base]){
                ans += self(self, lson[base], l, mid, x, y);
            }
            if (y > mid && rson[base]){
                ans += self(self, rson[base], mid + 1, r, x, y);
            }
            return ans;
        };
        return querynode(querynode, head[id], L, R, x, y);
    }
    // 查询x~y版本中新增的第k小元素
    // 即对于线段[x,y]求第k小元素
    int kthmin(int x, int y, int k){
        auto query = [&](auto &&self, int l, int r, int x, int y, int k){
            if (l == r)
                return l;
            int mid = l + (r - l >> 1);
            int ans = node[lson[y]] - node[lson[x]];
            if (ans >= k){
                return self(self, l, mid, lson[x], lson[y], k);
            }
            else{
                return self(self, mid + 1, r, rson[x], rson[y], k - ans);
            }
        };
        return query(query, L, R, head[x], head[y], k);
    }
    // 查询x~y版本中新增的第k大元素
    // 即对于线段[x,y]求第k大元素
    int kthmax(int x, int y, int k){
        int total = node[y] - node[x];
        return kthmin(x, y, total - k + 1); // 第k大 = 第(total-k+1)小
    }
};
```

## 可持久化01trietree

```cpp
struct TRIE{
    vector<array<int,2>> v;
    vector<int> head;
    vector<int> num;
    TRIE(){
        v.push_back({-1,-1});
        head.push_back(0);
        num.push_back(0);
        int now=0;
        for(int i=30;i>=0;i--){
            v[now][0]=v.size();
            num.push_back(1);
            v.push_back({-1,-1});
            now=v[now][0];
        }
    }
    void insert(int x){
        int base=head.back(),now=v.size();
        head.push_back(v.size());
        v.push_back({-1,-1});
        num.push_back(0);
        for(int i=30;i>=0;i--){
            int cnt=x>>i&1;
            if(base==-1) v[now][!cnt]=-1;
            else v[now][!cnt]=v[base][!cnt];
            v[now][cnt]=v.size();
            v.push_back({0,0});
            if(base==-1||v[base][cnt]==-1){
                num.push_back(1);
            }else{
                num.push_back(num[v[base][cnt]]+1);
            }
            now=v[now][cnt];
            if(base!=-1) base=v[base][cnt];
        }
    }
    int query(int l,int r,int x){
        int lnow,rnow;
        if(l-1<0) lnow=-1;
        else lnow=head[l-1];
        rnow=head[r];
        int ans=0;
        for(int i=30;i>=0;i--){
            int cnt=x>>i&1;
            if(v[rnow][!cnt]!=-1&&num[v[rnow][!cnt]]&&(lnow==-1||v[lnow][!cnt]==-1||num[v[rnow][!cnt]]>num[v[lnow][!cnt]])){
                ans|=1ll<<i;
                if(lnow!=-1) lnow=v[lnow][!cnt];
                rnow=v[rnow][!cnt];
            }else{
                if(lnow!=-1) lnow=v[lnow][cnt];
                rnow=v[rnow][cnt];
            }
        }
        return ans;
    }
};
```

## ST表

```cpp
struct ST {
    vector<vector<int>> dp;
    ST(int n, vector<int> &v) {
        dp.resize(20);
        for (int i = 0; i < 20; i++) {
            dp[i].resize(n + 1);
        }
        for (int i = 1; i <= n; i++) {
            dp[0][i] = v[i];
        }
        for (int i = 1; i <= 18; i++) {
            for (int j = 1; j + (1ll << i) - 1 <= n; j++) {
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j + (1ll << i - 1)]);
            }
        }
    }
    int query(int l, int r) {
        int k = __lg(r - l + 1);
        return max(dp[k][l], dp[k][r - (1ll << k) + 1]);
    }
};
```

## 权值线段树

```cpp
struct SegmentTree {
    struct edge {
        int sum, lson, rson;
    };
    int l, r;
    int cnt;
    vector<edge> node;
    SegmentTree(int l, int r) : l(l), r(r) {
        cnt = 1;
        node.push_back({0, 0, 0});
        node.push_back({0, 0, 0});
    }
    void pushup(int id, int l, int r) {
        node[id].sum = 0;
        if (node[id].lson)
            node[id].sum += node[node[id].lson].sum;
        if (node[id].rson)
            node[id].sum += node[node[id].rson].sum;
    }
    void update(int x, int delta) {
        auto upd = [&](auto &self, int id, int l, int r, int x, int delta) {
            if (l == r) {
                node[id].sum += delta;
                return;
            }
            int mid = l + (r - l >> 1);
            if (x <= mid) {
                if (!node[id].lson) {
                    node[id].lson = ++cnt;
                    node.push_back({0, 0, 0});
                }
                self(self, node[id].lson, l, mid, x, delta);
            } else {
                if (!node[id].rson) {
                    node[id].rson = ++cnt;
                    node.push_back({0, 0, 0});
                }
                self(self, node[id].rson, mid + 1, r, x, delta);
            }
            pushup(id, l, r);
        };
        upd(upd, 1, l, r, x, delta);
    }
    int query(int x, int y) {
        auto que = [&](auto &&self, int id, int l, int r, int x, int y) {
            if (x <= l && r <= y)
                return node[id].sum;
            int ans = 0;
            int mid = l + (r - l >> 1);
            if (x <= mid && node[id].lson) {
                ans += self(self, node[id].lson, l, mid, x, y);
            }
            if (y > mid && node[id].rson) {
                ans += self(self, node[id].rson, mid + 1, r, x, y);
            }
            return ans;
        };
        return que(que, 1, l, r, x, y);
    }
};
```

## 树套树

### 树状数组套权值线段树

区间kth等的问题，用树状数组来实现权值线段树的区间求和

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct SegmentTree{
    struct edge{
        int sum,lson,rson;
    };
    int l,r;
    int cnt;
    vector<edge> node;
    SegmentTree(int l,int r):l(l),r(r){
        cnt=1;
        node.push_back({0,0,0});
        node.push_back({0,0,0});
    }
    void pushup(int id,int l,int r){
        node[id].sum=0;
        if(node[id].lson) node[id].sum+=node[node[id].lson].sum;
        if(node[id].rson) node[id].sum+=node[node[id].rson].sum;
    }
    void update(int x,int delta){
        function<void(int,int,int,int,int)> upd=[&](int id,int l,int r,int x,int delta){
            //cerr<<l<<" "<<r<<"\n";
            if(l==r){
                node[id].sum+=delta;
                return;
            }
            int mid=l+(r-l>>1);
            if(x<=mid){
                if(!node[id].lson){
                    node[id].lson=++cnt;
                    node.push_back({0,0,0});
                }
                upd(node[id].lson,l,mid,x,delta);
            }else{
                if(!node[id].rson){
                    node[id].rson=++cnt;
                    node.push_back({0,0,0});
                }
                upd(node[id].rson,mid+1,r,x,delta);
            }
            pushup(id,l,r);
        };
        upd(1,l,r,x,delta);
    }
    int query(int x,int y){
        function<int(int,int,int,int,int)> que=[&](int id,int l,int r,int x,int y){
            if(x<=l&&r<=y) return node[id].sum;
            int ans=0;
            int mid=l+(r-l>>1);
            if(x<=mid&&node[id].lson){
                ans+=que(node[id].lson,l,mid,x,y);
            }
            if(y>mid&&node[id].rson){
                ans+=que(node[id].rson,mid+1,r,x,y);
            }
            return ans;
        };
        return que(1,l,r,x,y);
    }
};
struct BIT{
    vector<SegmentTree> tree;
    int n;
    inline int lowbit(int x){
        return x&(-x);
    }
    BIT(int n,int l,int r){
        this->n=n;
        tree.resize(n+1,SegmentTree(l,r));
    }
    void update(int x,int y,int delta){
        for(int i=x;i<=n;i+=lowbit(i)){
            tree[i].update(y,delta);
        }
    }
    int query(int x,int y,int a,int b){
        if(x>y) return 0;
        int ans=0;
        for(int i=y;i>=1;i-=lowbit(i)){
            ans+=tree[i].query(a,b);
        }
        for(int i=x-1;i>=1;i-=lowbit(i)){
            ans-=tree[i].query(a,b);
        }
        return ans;
    }
    int que(int l,int r,int x,int y,int k){
        if(x==y) return x;
        int mid=x+(y-x>>1);
        int num=query(l,r,x,mid);
        if(k<=num) return que(l,r,x,mid,k);
        else return que(l,r,mid+1,y,k-num);
    }
};
struct ss{
    char c;
    int x,y,z;
};
void solve(){
    int n,m;
    cin>>n>>m;
    vector<int> a(n+1);
    vector<int> num;
    vector<ss> query(m);
    function<int(int)> getid=[&](int x){
        return lower_bound(num.begin(),num.end(),x)-num.begin()+1;
    };
    for(int i=1;i<=n;i++){
        cin>>a[i];
        num.push_back(a[i]);
    }
    for(int i=0;i<m;i++){
        cin>>query[i].c;
        if(query[i].c=='Q'){
            cin>>query[i].x>>query[i].y>>query[i].z;
        }else{
            cin>>query[i].x>>query[i].y;
            num.push_back(query[i].y);
        }
    }
    sort(num.begin(),num.end());
    num.erase(unique(num.begin(),num.end()),num.end());
    BIT bit(n,1,num.size());
    for(int i=1;i<=n;i++) a[i]=getid(a[i]);
    for(int i=0;i<m;i++){
        if(query[i].c=='C'){
            query[i].y=getid(query[i].y);
        }
    }
    for(int i=1;i<=n;i++) bit.update(i,a[i],1);
    for(int i=0;i<m;i++){
        if(query[i].c=='Q'){
            cout<<num[bit.que(query[i].x,query[i].y,1,num.size(),query[i].z)-1]<<"\n";
        }else{
            int x=query[i].x;
            int y=query[i].y;
            bit.update(x,a[x],-1);
            bit.update(x,y,1);
            a[x]=y;
        }
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

### 线段树套平衡树

区间排名，单点修改，找区间前驱后继

```cpp
#include<bits/stdc++.h>
#include<bits/extc++.h>
using namespace std;
#define int long long
const int INF=1e18;
struct SegmentTree{
    struct edge{
        __gnu_pbds::tree<std::pair<int,int>,__gnu_pbds::null_type,
                    std::less<pair<int,int>>,__gnu_pbds::rb_tree_tag,
                    __gnu_pbds::tree_order_statistics_node_update>
                    tree;
    };
    vector<edge> node;
    int n;
    int clock;
    SegmentTree(int n):n(n){
        node.resize((n<<2)+5);
        clock=0;
    }
    SegmentTree(){}
    void init(vector<pair<int,int>> &v){
        function<void(int,int,int)> buildtree=[&](int id,int l,int r){
            for(int i=l;i<=r;i++){
                node[id].tree.insert(v[i]);
            }
            if(l==r) return;
            int mid=l+(r-l>>1);
            buildtree(id<<1,l,mid);
            buildtree(id<<1|1,mid+1,r);
        };
        buildtree(1,1,n);
    }
    SegmentTree(int n,vector<pair<int,int>> &v):n(n){
        node.resize((n<<2)+5);
        init(v);
    }
    void update(int w,pair<int,int> x,pair<int,int> y){
        function<void(int,int,int,int,pair<int,int>,pair<int,int>)> upd=[&](int id,int l,int r,int w,pair<int,int> x,pair<int,int> y){
            node[id].tree.erase(x);
            node[id].tree.insert(y);
            if(l==r){
                return;
            }
            int mid=l+(r-l>>1);
            if(w<=mid) upd(id<<1,l,mid,w,x,y);
            else upd(id<<1|1,mid+1,r,w,x,y);
        };
        upd(1,1,n,w,x,y);
    }
    int querynum(int id,int l,int r,int x,int y,int t){
        if(x<=l&&r<=y) return (int)node[id].tree.order_of_key({t,0});
        int ans=0;
        int mid=l+(r-l>>1);
        if(x<=mid){
            ans+=querynum(id<<1,l,mid,x,y,t);
        }
        if(y>mid){
            ans+=querynum(id<<1|1,mid+1,r,x,y,t);
        }
        return ans;
    };
    int queryrnk(int l,int r,int x){
        return querynum(1,1,n,l,r,x)+1;
    }
    int querybyrnk(int x,int y,int k){
        int l=0,r=1e8,ans=-1;
        while(l<=r){
            int mid=l+(r-l>>1);
            int num=querynum(1,1,n,x,y,mid);
            if(num<=k-1){
                ans=mid;
                l=mid+1;
            }else r=mid-1;
        }
        return ans;
    }
    int querypre(int x,int y,int k){
        function<int(int,int,int,int,int,int)> query=[&](int id,int l,int r,int x,int y,int t){
            if(x<=l&&r<=y){
                auto it=node[id].tree.lower_bound({t,0});
                if(it==node[id].tree.begin()) return -2147483647ll;
                --it;
                return it->first;
            }
            int mid=l+(r-l>>1);
            int ans=-2147483647ll;
            if(x<=mid) ans=max(ans,query(id<<1,l,mid,x,y,t));
            if(y>mid) ans=max(ans,query(id<<1|1,mid+1,r,x,y,t));
            return ans;
        };
        return query(1,1,n,x,y,k);
    }
    int querynxt(int x,int y,int k){
        function<int(int,int,int,int,int,int)> query=[&](int id,int l,int r,int x,int y,int t){
            if(x<=l&&r<=y){
                auto it=node[id].tree.upper_bound({t,INF});
                if(it==node[id].tree.end()) return 2147483647ll;
                return it->first;
            }
            int mid=l+(r-l>>1);
            int ans=2147483647ll;
            if(x<=mid) ans=min(ans,query(id<<1,l,mid,x,y,t));
            if(y>mid) ans=min(ans,query(id<<1|1,mid+1,r,x,y,t));
            return ans;
        };
        return query(1,1,n,x,y,k);
    }
};
void solve(){
    int n,m;
    cin>>n>>m;
    vector<pair<int,int>> a(n+1);
    int clock=0;
    for(int i=1;i<=n;i++){
        cin>>a[i].first;
        a[i].second=++clock;
    }
    SegmentTree tree(n,a);
    while(m--){
        int op;
        cin>>op;
        if(op==1){
            int l,r,k;
            cin>>l>>r>>k;
            cout<<tree.queryrnk(l,r,k)<<"\n";
        }else if(op==2){
            int l,r,k;
            cin>>l>>r>>k;
            cout<<tree.querybyrnk(l,r,k)<<"\n";
        }else if(op==3){
            int pos,k;
            cin>>pos>>k;
            tree.update(pos,a[pos],{k,++clock});
            a[pos]={k,clock};
        }else if(op==4){
            int l,r,k;
            cin>>l>>r>>k;
            cout<<tree.querypre(l,r,k)<<"\n";
        }else{
            int l,r,k;
            cin>>l>>r>>k;
            cout<<tree.querynxt(l,r,k)<<"\n";
        }
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 猫树

$O(nlogn)$次合并，查询时$O(1)$次合并查询

具体做法是，每个节点，$l$ ,$mid$ ,$r$ ,存储 $mid$ 往前的后缀贡献，和 $mid$ 往后的前缀贡献

查询区间 $[l,r]$，则只需要在他们的 $lca$ 上查找前缀后缀即可 $O(1)$ 查询

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct CatTree{
    int n;
    vector<int> pos;//[i,i]所在的位置
    vector<pair<int,int>> lr;
    vector<array<vector<int>,2>> maxn;//某个区间，从mid分开的后缀前缀贡献
    int lca(int x,int y){
        int xx=__builtin_clz(x);
        int yy=__builtin_clz(y);
        if(xx<yy) x>>=yy-xx;
        else y>>=xx-yy;
        return x>>(__lg(x^y)+1);
    }
    CatTree(int n,vector<int> &v):n(n),pos(n+1,0),maxn((n<<2)+10),lr((n<<2)+10){
        auto buildtree=[&](auto &&self,int id,int l,int r){
            int mid=l+(r-l>>1);
            maxn[id][0].reserve(mid-l+2);
            maxn[id][1].reserve(r-mid+1);
            maxn[id][0].push_back(-1e18);
            maxn[id][1].push_back(-1e18);
            lr[id]={l,r};
            if(l==r){
                pos[l]=id;
                maxn[id][0].push_back(v[l]);
                return;
            }
            for(int i=mid;i>=l;i--){
                maxn[id][0].push_back(max(maxn[id][0].back(),v[i]));
            }
            for(int i=mid+1;i<=r;i++){
                maxn[id][1].push_back(max(maxn[id][1].back(),v[i]));
            }
            self(self,id<<1,l,mid);
            self(self,id<<1|1,mid+1,r);
        };
        buildtree(buildtree,1,1,n);
    }
    int query(int x,int y){
        int l=lca(pos[x],pos[y]);
        int mid=(lr[l].first+lr[l].second)>>1;
        return max(maxn[l][0][mid-x+1],maxn[l][1][y-mid]);
    }
};
void solve(){
    int n,m;
    cin>>n>>m;
    vector<int> v(n+1);
    for(int i=1;i<=n;i++){
        cin>>v[i];
    }
    CatTree cattree(n,v);
    while(m--){
        int x,y;
        cin>>x>>y;
        cout<<cattree.query(x,y)<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 可删堆

```c++
template <class T, class Cmp=less<T>>
class DeletableHeap {
    priority_queue<T, std::vector<T>, Cmp> items{};
    priority_queue<T, std::vector<T>, Cmp> trash{};
    void sync() {
        while (!trash.empty() and items.top() == trash.top()) {
            items.pop();
            trash.pop();
        }
    }
public:
    bool empty(){
        sync();
        return items.empty();
    }
    int size() {
        assert(items.size() >= trash.size());
        return items.size() - trash.size();
    }
    void push(T x) {
        items.push(x);
    }
    void erase(T x) {
        trash.push(x);
    }
    T top() {
        sync();
        return items.top();
    }
    void pop() {
        sync();
        items.pop();
    }
};
```

## LCT

```cpp
struct LCT{
    struct Node{
        int fa = 0, lson = 0, rson = 0; // 父、左、右
        int val = 0;                    // 节点权值
        int path_sum = 0;               // 路径汇总（异或）
        bool rev_flag = false;          // 反转标记
        // 若要维护其他信息，可在此添加成员变量，如：int max_val, min_val, size, add_lazy;
    };
    vector<Node> tree;
    LCT(int n = 0) : tree(n + 1) {}
    // 判断节点x是否不是其所在Splay的根
    bool not_root(int x){
        int f = tree[x].fa;
        return f && (tree[f].lson == x || tree[f].rson == x);
    }
    // ========== 信息维护核心：更新节点信息 ==========
    void push_up(int x){
        int l = tree[x].lson, r = tree[x].rson;
        int lv = l ? tree[l].path_sum : 0;
        int rv = r ? tree[r].path_sum : 0;
        tree[x].path_sum = lv ^ rv ^ tree[x].val;
        /* 其他常见信息维护示例：
        // 维护路径和：
        tree[x].path_sum = tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val;

        // 维护路径最大值（需初始化max_val为负无穷，注意处理叶子节点）：
        tree[x].max_val = max({tree[tree[x].lson].max_val, tree[tree[x].rson].max_val, tree[x].val});

        // 维护路径最小值（需初始化min_val为正无穷）：
        tree[x].min_val = min({tree[tree[x].lson].min_val, tree[tree[x].rson].min_val, tree[x].val});

        // 维护子树大小（节点数）：
        tree[x].size = tree[tree[x].lson].size + tree[tree[x].rson].size + 1;

        // 同时维护多种信息（例如求和与求最大值）：
        tree[x].path_sum = tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val;
        tree[x].max_val = max({tree[tree[x].lson].max_val, tree[tree[x].rson].max_val, tree[x].val});
        */
    }
    // 翻转节点x的左右子树
    void push_rev(int x){
        if (!x)
            return;
        swap(tree[x].lson, tree[x].rson);
        tree[x].rev_flag = !tree[x].rev_flag;
    }
    // 下传标记：此处主要处理翻转标记
    void push_down(int x){
        if (!x)
            return;
        if (tree[x].rev_flag){
            if (tree[x].lson)
                push_rev(tree[x].lson);
            if (tree[x].rson)
                push_rev(tree[x].rson);
            tree[x].rev_flag = false;
        }
        // 示例：处理区间加懒标记 (若存在名为add_lazy的懒标记)
        // if (tree[x].add_lazy != 0) {
        //     if (tree[x].lson) { /* 更新tree[tree[x].lson].val和tree[tree[x].lson].add_lazy */ }
        //     if (tree[x].rson) { /* 更新tree[tree[x].rson].val和tree[tree[x].rson].add_lazy */ }
        //     tree[x].add_lazy = 0;
        // }
    }
    // 旋转操作（Splay基础）
    void rotate(int x){
        int y = tree[x].fa, z = tree[y].fa;
        bool is_x_right = (tree[y].rson == x);
        if (not_root(y)){
            if (tree[z].lson == y)
                tree[z].lson = x;
            else
                tree[z].rson = x;
        }
        tree[x].fa = z;
        if (is_x_right){
            tree[y].rson = tree[x].lson;
            if (tree[x].lson)
                tree[tree[x].lson].fa = y;
            tree[x].lson = y;
            tree[y].fa = x;
        }
        else{
            // x 是 y 的左儿子
            tree[y].lson = tree[x].rson;
            if (tree[x].rson)
                tree[tree[x].rson].fa = y;
            tree[x].rson = y;
            tree[y].fa = x;
        }
        push_up(y);
        push_up(x);
    }
    // Splay操作：将x旋转到当前Splay的根
    void splay(int x){
        if (!x)
            return;
        vector<int> stk;
        int y = x;
        stk.push_back(y);
        while (not_root(y)){
            y = tree[y].fa;
            stk.push_back(y);
        }
        for (int i = (int)stk.size() - 1; i >= 0; --i)
            push_down(stk[i]);

        while (not_root(x)){
            int y = tree[x].fa, z = tree[y].fa;
            if (not_root(y)){
                if ((tree[y].lson == x) ^ (tree[z].lson == y))
                    rotate(x);
                else
                    rotate(y);
            }
            rotate(x);
        }
    }
    // access：将从根到 x 的实链打通，使 x 在其代表树中位于最右（深度最大）
    void access(int x){
        int v = 0;
        for (int u = x; u; u = tree[u].fa){
            splay(u);
            tree[u].rson = v;
            push_up(u);
            v = u;
        }
    }
    // 将 x 设为原树的根
    void make_root(int x){
        access(x);
        splay(x);
        push_rev(x);
    }
    // 找到所在原树的根（用于判断连通性）
    int find_root(int x){
        access(x);
        splay(x);
        while (tree[x].lson){
            push_down(x);
            x = tree[x].lson;
        }
        splay(x);
        return x;
    }
    // 分离并提取 x -> y 的路径，执行后 y 是这条路径对应 Splay 的根
    void split(int x, int y){
        make_root(x);
        access(y);
        splay(y);
    }
    // 连接 x 和 y（在原树中加边）
    bool link(int x, int y){
        make_root(x);
        if (find_root(y) != x){
            tree[x].fa = y;
            return true;
        }
        return false;
    }
    // 断开 x 和 y 之间的边（在原树中删边）
    bool cut(int x, int y){
        make_root(x);
        if (find_root(y) == x && tree[y].fa == x && tree[y].lson == 0){
            tree[y].fa = 0;
            tree[x].rson = 0;
            push_up(x);
            return true;
        }
        return false;
    }
    // 修改节点 x 的权值
    void update_node(int x, int new_val){
        splay(x);
        tree[x].val = new_val;
        push_up(x);
    }
    // 查询 x 到 y 路径上的信息（此处为异或和）
    int query_path(int x, int y){
        split(x, y);
        return tree[y].path_sum;
    }
    // 判断 x 和 y 是否连通
    bool is_connected(int x, int y){
        return find_root(x) == find_root(y);
    }
};
```

### 区间

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int MOD=51061;
struct LCT{
    struct Node{
        int fa = 0, lson = 0, rson = 0; // 父、左、右
        int val = 0;                    // 节点权值
        int path_sum=0;
        int size = 0;               // 路径汇总（异或）
        bool rev_flag = false;          // 反转标记
        int add_lazy=0;
        int mul_lazy=1;
        // 若要维护其他信息，可在此添加成员变量，如：int max_val, min_val, size, add_lazy;
    };
    vector<Node> tree;
    LCT(int n = 0) : tree(n + 1) {}
    // 判断节点x是否不是其所在Splay的根
    bool not_root(int x){
        int f = tree[x].fa;
        return f && (tree[f].lson == x || tree[f].rson == x);
    }
    // ========== 信息维护核心：更新节点信息 ==========
    void push_up(int x){
        int l = tree[x].lson, r = tree[x].rson;
        int lv = l ? tree[l].size : 0;
        int rv = r ? tree[r].size : 0;
        tree[x].size = tree[tree[x].lson].size + tree[tree[x].rson].size + 1;
        tree[x].path_sum = (tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val)%MOD;
        //tree[x].path_sum = lv ^ rv ^ tree[x].val;
        /* 其他常见信息维护示例：
        // 维护路径和：
        tree[x].path_sum = tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val;

        // 维护路径最大值（需初始化max_val为负无穷，注意处理叶子节点）：
        tree[x].max_val = max({tree[tree[x].lson].max_val, tree[tree[x].rson].max_val, tree[x].val});

        // 维护路径最小值（需初始化min_val为正无穷）：
        tree[x].min_val = min({tree[tree[x].lson].min_val, tree[tree[x].rson].min_val, tree[x].val});

        // 维护子树大小（节点数）：
        tree[x].size = tree[tree[x].lson].size + tree[tree[x].rson].size + 1;

        // 同时维护多种信息（例如求和与求最大值）：
        tree[x].path_sum = tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val;
        tree[x].max_val = max({tree[tree[x].lson].max_val, tree[tree[x].rson].max_val, tree[x].val});
        */
    }
    // 翻转节点x的左右子树
    void push_rev(int x){
        if (!x)
            return;
        swap(tree[x].lson, tree[x].rson);
        tree[x].rev_flag = !tree[x].rev_flag;
    }
    // 下传标记：此处主要处理翻转标记
    void push_down(int x){
        if (!x)
            return;
        if (tree[x].rev_flag){
            if (tree[x].lson)
                push_rev(tree[x].lson);
            if (tree[x].rson)
                push_rev(tree[x].rson);
            tree[x].rev_flag = false;
        }
        // 示例：处理区间加懒标记 (若存在名为add_lazy的懒标记)
        // if (tree[x].add_lazy != 0) {
        //     if (tree[x].lson) { /* 更新tree[tree[x].lson].val和tree[tree[x].lson].add_lazy */ }
        //     if (tree[x].rson) { /* 更新tree[tree[x].rson].val和tree[tree[x].rson].add_lazy */ }
        //     tree[x].add_lazy = 0;
        // }
        if(tree[x].mul_lazy!=1){
            if(tree[x].lson){
                (tree[tree[x].lson].val*=tree[x].mul_lazy)%=MOD;
                (tree[tree[x].lson].path_sum*=tree[x].mul_lazy)%=MOD;
                (tree[tree[x].lson].mul_lazy*=tree[x].mul_lazy)%=MOD;
                (tree[tree[x].lson].add_lazy*=tree[x].mul_lazy)%=MOD;
            }
            if(tree[x].rson){
                (tree[tree[x].rson].val*=tree[x].mul_lazy)%=MOD;
                (tree[tree[x].rson].path_sum*=tree[x].mul_lazy)%=MOD;
                (tree[tree[x].rson].mul_lazy*=tree[x].mul_lazy)%=MOD;
                (tree[tree[x].rson].add_lazy*=tree[x].mul_lazy)%=MOD;
            }
            tree[x].mul_lazy=1;
        }
        if (tree[x].add_lazy != 0) {
            if (tree[x].lson) {
                (tree[tree[x].lson].val+=tree[x].add_lazy)%=MOD;
                (tree[tree[x].lson].path_sum+=tree[x].add_lazy*tree[tree[x].lson].size)%=MOD;
                (tree[tree[x].lson].add_lazy+=tree[x].add_lazy)%=MOD;
                /* 更新tree[tree[x].lson].val和tree[tree[x].lson].add_lazy */ 
            }
            if (tree[x].rson) {
                (tree[tree[x].rson].val+=tree[x].add_lazy)%=MOD;
                (tree[tree[x].rson].path_sum+=tree[x].add_lazy*tree[tree[x].rson].size)%=MOD;
                (tree[tree[x].rson].add_lazy+=tree[x].add_lazy)%=MOD;
                 /* 更新tree[tree[x].rson].val和tree[tree[x].rson].add_lazy */ 
            }
            tree[x].add_lazy = 0;
        }
    }
    // 旋转操作（Splay基础）
    void rotate(int x){
        int y = tree[x].fa, z = tree[y].fa;
        bool is_x_right = (tree[y].rson == x);
        if (not_root(y)){
            if (tree[z].lson == y)
                tree[z].lson = x;
            else
                tree[z].rson = x;
        }
        tree[x].fa = z;
        if (is_x_right){
            tree[y].rson = tree[x].lson;
            if (tree[x].lson)
                tree[tree[x].lson].fa = y;
            tree[x].lson = y;
            tree[y].fa = x;
        }
        else{
            // x 是 y 的左儿子
            tree[y].lson = tree[x].rson;
            if (tree[x].rson)
                tree[tree[x].rson].fa = y;
            tree[x].rson = y;
            tree[y].fa = x;
        }
        push_up(y);
        push_up(x);
    }
    // Splay操作：将x旋转到当前Splay的根
    void splay(int x){
        if (!x)
            return;
        vector<int> stk;
        int y = x;
        stk.push_back(y);
        while (not_root(y)){
            y = tree[y].fa;
            stk.push_back(y);
        }
        for (int i = (int)stk.size() - 1; i >= 0; --i)
            push_down(stk[i]);

        while (not_root(x)){
            int y = tree[x].fa, z = tree[y].fa;
            if (not_root(y)){
                if ((tree[y].lson == x) ^ (tree[z].lson == y))
                    rotate(x);
                else
                    rotate(y);
            }
            rotate(x);
        }
    }
    // access：将从根到 x 的实链打通，使 x 在其代表树中位于最右（深度最大）
    void access(int x){
        int v = 0;
        for (int u = x; u; u = tree[u].fa){
            splay(u);
            tree[u].rson = v;
            push_up(u);
            v = u;
        }
    }
    // 将 x 设为原树的根
    void make_root(int x){
        access(x);
        splay(x);
        push_rev(x);
    }
    // 找到所在原树的根（用于判断连通性）
    int find_root(int x){
        access(x);
        splay(x);
        while (tree[x].lson){
            push_down(x);
            x = tree[x].lson;
        }
        splay(x);
        return x;
    }
    // 分离并提取 x -> y 的路径，执行后 y 是这条路径对应 Splay 的根
    void split(int x, int y){
        make_root(x);
        access(y);
        splay(y);
    }
    // 连接 x 和 y（在原树中加边）
    bool link(int x, int y){
        make_root(x);
        if (find_root(y) != x){
            tree[x].fa = y;
            return true;
        }
        return false;
    }
    // 断开 x 和 y 之间的边（在原树中删边）
    bool cut(int x, int y){
        make_root(x);
        if (find_root(y) == x && tree[y].fa == x && tree[y].lson == 0){
            tree[y].fa = 0;
            tree[x].rson = 0;
            push_up(x);
            return true;
        }
        return false;
    }
    // 修改节点 x 的权值
    // void update_node(int x, int new_val){
    //     splay(x);
    //     tree[x].val = new_val;
    //     push_up(x);
    // }
    void update_node_add(int x,int y,int delta){
        split(x,y);
        (tree[y].val+=delta)%=MOD;
        (tree[y].path_sum+=delta*tree[y].size%MOD)%=MOD;
        (tree[y].add_lazy+=delta)%=MOD;
        push_up(y);
    }
    void update_node_mul(int x,int y,int delta){
        split(x,y);
        (tree[y].val*=delta)%=MOD;
        (tree[y].path_sum*=delta)%MOD;
        (tree[y].mul_lazy*=delta)%=MOD;
        (tree[y].add_lazy*=delta)%=MOD;
        push_up(y);
    }
    // 查询 x 到 y 路径上的信息（此处为异或和）
    int query_path(int x, int y){
        split(x, y);
        return tree[y].path_sum;
    }
    // 判断 x 和 y 是否连通
    bool is_connected(int x, int y){
        return find_root(x) == find_root(y);
    }
};
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int n,q;
    cin>>n>>q;
    LCT lct(n);
    for(int i=1;i<n;i++){
        int x,y;
        cin>>x>>y;
        lct.link(x,y);
    }
    for(int i=1;i<=n;i++){
        lct.update_node_add(i,i,1);
    }
    while(q--){
        char op;
        cin>>op;
        if(op=='+'){
            int a,b,c;
            cin>>a>>b>>c;
            lct.update_node_add(a,b,c);
        }else if(op=='-'){
            int a,b,c,d;
            cin>>a>>b>>c>>d;
            lct.cut(a,b);
            lct.link(c,d);
        }else if(op=='*'){
            int a,b,c;
            cin>>a>>b>>c;
            lct.update_node_mul(a,b,c);
        }else{
            int a,b;
            cin>>a>>b;
            cout<<lct.query_path(a,b)<<"\n";
        }
    }
}
```

### 最值and位置

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int INF=1e18;
struct LCT{
    struct Node{
        int fa = 0, lson = 0, rson = 0; // 父、左、右
        int val = 0;                    // 节点权值
        //int path_sum = 0;               // 路径汇总（异或）
        pair<int,int> maxn;
        bool rev_flag = false;          // 反转标记
        // 若要维护其他信息，可在此添加成员变量，如：int max_val, min_val, size, add_lazy;
    };
    vector<Node> tree;
    LCT(int n = 0) : tree(n + 1) {}
    // 判断节点x是否不是其所在Splay的根
    bool not_root(int x){
        int f = tree[x].fa;
        return f && (tree[f].lson == x || tree[f].rson == x);
    }
    // ========== 信息维护核心：更新节点信息 ==========
    void push_up(int x){
        int l = tree[x].lson, r = tree[x].rson;
        auto lv = l ? tree[l].maxn : make_pair(-INF,0ll);
        auto rv = r ? tree[r].maxn : make_pair(-INF,0ll);
        tree[x].maxn={tree[x].val,x};
        tree[x].maxn=max({tree[x].maxn,lv,rv});
        //tree[x].path_sum = lv ^ rv ^ tree[x].val;
        /* 其他常见信息维护示例：
        // 维护路径和：
        tree[x].path_sum = tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val;

        // 维护路径最大值（需初始化max_val为负无穷，注意处理叶子节点）：
        tree[x].max_val = max({tree[tree[x].lson].max_val, tree[tree[x].rson].max_val, tree[x].val});

        // 维护路径最小值（需初始化min_val为正无穷）：
        tree[x].min_val = min({tree[tree[x].lson].min_val, tree[tree[x].rson].min_val, tree[x].val});

        // 维护子树大小（节点数）：
        tree[x].size = tree[tree[x].lson].size + tree[tree[x].rson].size + 1;

        // 同时维护多种信息（例如求和与求最大值）：
        tree[x].path_sum = tree[tree[x].lson].path_sum + tree[tree[x].rson].path_sum + tree[x].val;
        tree[x].max_val = max({tree[tree[x].lson].max_val, tree[tree[x].rson].max_val, tree[x].val});
        */
    }
    // 翻转节点x的左右子树
    void push_rev(int x){
        if (!x)
            return;
        swap(tree[x].lson, tree[x].rson);
        tree[x].rev_flag = !tree[x].rev_flag;
    }
    // 下传标记：此处主要处理翻转标记
    void push_down(int x){
        if (!x)
            return;
        if (tree[x].rev_flag){
            if (tree[x].lson)
                push_rev(tree[x].lson);
            if (tree[x].rson)
                push_rev(tree[x].rson);
            tree[x].rev_flag = false;
        }
        // 示例：处理区间加懒标记 (若存在名为add_lazy的懒标记)
        // if (tree[x].add_lazy != 0) {
        //     if (tree[x].lson) { /* 更新tree[tree[x].lson].val和tree[tree[x].lson].add_lazy */ }
        //     if (tree[x].rson) { /* 更新tree[tree[x].rson].val和tree[tree[x].rson].add_lazy */ }
        //     tree[x].add_lazy = 0;
        // }
    }
    // 旋转操作（Splay基础）
    void rotate(int x){
        int y = tree[x].fa, z = tree[y].fa;
        bool is_x_right = (tree[y].rson == x);
        if (not_root(y)){
            if (tree[z].lson == y)
                tree[z].lson = x;
            else
                tree[z].rson = x;
        }
        tree[x].fa = z;
        if (is_x_right){
            tree[y].rson = tree[x].lson;
            if (tree[x].lson)
                tree[tree[x].lson].fa = y;
            tree[x].lson = y;
            tree[y].fa = x;
        }
        else{
            // x 是 y 的左儿子
            tree[y].lson = tree[x].rson;
            if (tree[x].rson)
                tree[tree[x].rson].fa = y;
            tree[x].rson = y;
            tree[y].fa = x;
        }
        push_up(y);
        push_up(x);
    }
    // Splay操作：将x旋转到当前Splay的根
    void splay(int x){
        if (!x)
            return;
        vector<int> stk;
        int y = x;
        stk.push_back(y);
        while (not_root(y)){
            y = tree[y].fa;
            stk.push_back(y);
        }
        for (int i = (int)stk.size() - 1; i >= 0; --i)
            push_down(stk[i]);

        while (not_root(x)){
            int y = tree[x].fa, z = tree[y].fa;
            if (not_root(y)){
                if ((tree[y].lson == x) ^ (tree[z].lson == y))
                    rotate(x);
                else
                    rotate(y);
            }
            rotate(x);
        }
    }
    // access：将从根到 x 的实链打通，使 x 在其代表树中位于最右（深度最大）
    void access(int x){
        int v = 0;
        for (int u = x; u; u = tree[u].fa){
            splay(u);
            tree[u].rson = v;
            push_up(u);
            v = u;
        }
    }
    // 将 x 设为原树的根
    void make_root(int x){
        access(x);
        splay(x);
        push_rev(x);
    }
    // 找到所在原树的根（用于判断连通性）
    int find_root(int x){
        access(x);
        splay(x);
        while (tree[x].lson){
            push_down(x);
            x = tree[x].lson;
        }
        splay(x);
        return x;
    }
    // 分离并提取 x -> y 的路径，执行后 y 是这条路径对应 Splay 的根
    void split(int x, int y){
        make_root(x);
        access(y);
        splay(y);
    }
    // 连接 x 和 y（在原树中加边）
    bool link(int x, int y){
        make_root(x);
        if (find_root(y) != x){
            tree[x].fa = y;
            return true;
        }
        return false;
    }
    // 断开 x 和 y 之间的边（在原树中删边）
    bool cut(int x, int y){
        make_root(x);
        if (find_root(y) == x && tree[y].fa == x && tree[y].lson == 0){
            tree[y].fa = 0;
            tree[x].rson = 0;
            push_up(x);
            return true;
        }
        return false;
    }
    // 修改节点 x 的权值
    void update_node(int x, int new_val){
        splay(x);
        tree[x].val = new_val;
        tree[x].maxn={new_val,x};
        push_up(x);
    }
    // 查询 x 到 y 路径上的信息（此处为异或和）
    pair<int,int> query_path(int x, int y){
        split(x, y);
        return tree[y].maxn;
    }
    // 判断 x 和 y 是否连通
    bool is_connected(int x, int y){
        return find_root(x) == find_root(y);
    }
};
struct edge{
    int x,y,a,b;
};
void solve(){
    int n,m;
    cin>>n>>m;
    LCT lct(n+m);
    vector<edge> v(m+1);
    for(int i=1;i<=m;i++){
        cin>>v[i].x>>v[i].y>>v[i].a>>v[i].b;
    }
    sort(v.begin()+1,v.end(),[&](const auto &a,const auto &b){
        return a.a<b.a;
    });
    int ans=INF;
    for(int i=1;i<=m;i++){
        auto [x,y,a,b]=v[i];
        if(!lct.is_connected(x,y)){
            lct.link(x,i+n);
            lct.link(y,i+n);
            lct.update_node(i+n,b);
        }else{
            auto [maxn,pos]=lct.query_path(x,y);
            if(b<=maxn){
                lct.cut(v[pos-n].x,pos);
                lct.cut(v[pos-n].y,pos);
                lct.link(x,i+n);
                lct.link(y,i+n);
                lct.update_node(i+n,b);
            }
        }
        if(lct.is_connected(1,n)){
            ans=min(ans,a+lct.query_path(1,n).first);
        }
    }
    if(ans==INF) cout<<-1<<"\n";
    else cout<<ans<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## FHQ

按照值split

```cpp
mt19937_64 rnd(time(0));
struct fhq{
    struct edge{
        int l,r,val,sz,rndnum;
        edge(int val):l(0),r(0),val(val),sz(1),rndnum(rnd()){}
    };
    vector<edge> node;
    int root;
    fhq(){
        root=0;
        node.emplace_back(0);
        node.front().sz=0;
    }
    int newnode(int val){
        node.emplace_back(val);
        return node.size()-1;
    }
    void pushup(int id){
        node[id].sz=node[node[id].l].sz+node[node[id].r].sz+1;
    }
    //分成值域在[-INF,val] (val,INF)的两棵子树
    pair<int,int> split(int id,int val){
        if(!id) return {0,0};
        if(node[id].val<=val){
            auto [x,y]=split(node[id].r,val);
            node[id].r=x;
            pushup(id);
            return {id,y};
        }else{
            auto [x,y]=split(node[id].l,val);
            node[id].l=y;
            pushup(id);
            return {x,id};
        }
    }
    int merge(int x,int y){
        if(!x||!y) return x+y;
        if(node[x].rndnum<node[y].rndnum){
            node[x].r=merge(node[x].r,y);
            pushup(x);
            return x;
        }else{
            node[y].l=merge(x,node[y].l);
            pushup(y);
            return y;
        }
    }
    //加入元素val
    void insert(int val){
        auto [x,y]=split(root,val);
        int z=newnode(val);
        root=merge(merge(x,z),y);
    }
    //删除元素val
    void del(int val){
        int x,y,z;
        auto [ll,r]=split(root,val);
        auto [l,mid]=split(ll,val-1);
        mid=merge(node[mid].l,node[mid].r);
        root=merge(merge(l,mid),r);
    }
    //求val的排名
    int getrnk(int val){
        auto [x,y]=split(root,val-1);
        int ans=node[x].sz+1;
        root=merge(x,y);
        return ans;
    }
    //求第k小元素
    int topk(int k,int id=-1){
        if(id==-1) id=root;
        int lsz=node[node[id].l].sz;
        if(k==lsz+1){
            return node[id].val;
        }else if(k<=lsz){
            return topk(k,node[id].l);
        }else{
            return topk(k-lsz-1,node[id].r);
        }
    }
    //获取前驱
    int getpre(int val){
        auto [x,y]=split(root,val-1);
        int ans=topk(node[x].sz,x);
        root=merge(x,y);
        return ans;
    }
    //获取后继
    int getnxt(int val){
        auto [x,y]=split(root,val);
        int ans=topk(1,y);
        root=merge(x,y);
        return ans;
    }
};
```

按照下标split，懒标记

```cpp
mt19937 rnd(time(0));
struct fhq{
    struct edge{
        int l,r,val,sz,rndnum,rev;
        edge(int val):l(0),r(0),val(val),sz(1),rndnum(rnd()),rev(0){}
    };
    vector<edge> node;
    int root;
    fhq(){
        root=0;
        node.emplace_back(0);
        node.front().sz=0;
    }
    int newnode(int val){
        node.emplace_back(val);
        return node.size()-1;
    }
    void pushup(int id){
        node[id].sz=node[node[id].l].sz+node[node[id].r].sz+1;
    }
    void doreverse(int x){
        swap(node[x].l,node[x].r);
    }
    void pushdown(int id){
        if(node[id].rev){
            node[node[id].l].rev^=1;
            node[node[id].r].rev^=1;
            doreverse(node[id].l);
            doreverse(node[id].r);
            node[id].rev=0;
        }
    }
    pair<int,int> split(int id,int val){
        if(!id) return {0,0};
        pushdown(id);
        if(node[node[id].l].sz+1<=val){
            auto [x,y]=split(node[id].r,val-(node[node[id].l].sz+1));
            node[id].r=x;
            pushup(id);
            return {id,y};
        }else{
            auto [x,y]=split(node[id].l,val);
            node[id].l=y;
            pushup(id);
            return {x,id};
        }
    }
    int merge(int x,int y){
        if(!x||!y) return x+y;
        if(node[x].rndnum<node[y].rndnum){
            pushdown(x);
            node[x].r=merge(node[x].r,y);
            pushup(x);
            return x;
        }else{
            pushdown(y);
            node[y].l=merge(x,node[y].l);
            pushup(y);
            return y;
        }
    }
    //加入元素val
    void insert(int val){
        auto [x,y]=split(root,val);
        int z=newnode(val);
        root=merge(merge(x,z),y);
    }
    void reve(int x,int y){
        auto [ll,r]=split(root,y);
        auto [l,mid]=split(ll,x-1);
        node[mid].rev^=1;
        doreverse(mid);
        merge(merge(l,mid),r);
    }
};
```

# 字符串

## 序列自动机

$nxt[i][j]$表示从第i个位置开始，字符串j出现的第一个位置

```cpp
struct SubsequenceAutomaton{
    string s;
    int n;
    vector<array<int,26>> nxt;
    SubsequenceAutomaton(string ss):s(ss){
        n=s.size();
        nxt.resize(n+1);
        for(int i=0;i<26;i++) nxt[n][i]=-1;
        for(int i=n-1;i>=0;i--){
            nxt[i]=nxt[i+1];
            if(s[i]>='a'&&s[i]<='z') nxt[i][s[i]-'a']=i;
        }
    }
    inline int query(int pos,string t){
        if(pos>=n) return -1;
        for(int i=0;i<t.size();i++){
            int p=t[i]-'a';
            if(nxt[pos][p]==-1) return -1;
            pos=nxt[pos][p];
            if(i!=t.size()-1) pos++;
        }
        return pos;
    }
};
```

## AC自动机

多模式串匹配

```cpp
struct ACAutomaton{
    vector<array<int,26>> tree;
    vector<int> fail,ed;
    int cnt;
    void insert(const string &s){
        int now=0;
        for(auto &c:s){
            if(!tree[now][c-'a']){
                tree[now][c-'a']=++cnt;
                fail.emplace_back(0);
                ed.emplace_back(0);
                tree.emplace_back();
            }
            now=tree[now][c-'a'];
        }
        ed[now]++;
    }
    void build(){
        queue<int> q;
        for(int i=0;i<26;i++){
            if(tree[0][i]) q.push(tree[0][i]);
        }
        while(!q.empty()){
            int x=q.front();
            q.pop();
            for(int i=0;i<26;i++){
                if(tree[x][i]){
                    fail[tree[x][i]]=tree[fail[x]][i];
                    q.push(tree[x][i]);
                }else{
                    tree[x][i]=tree[fail[x]][i];
                }
            }
        }
    }
    int query(const string &s){
        int ans=0,now=0;
        for(int i=0;i<s.size();i++){
            now=tree[now][s[i]-'a'];
            for(int j=now;j&&ed[j]!=-1;j=fail[j]){
                ans+=ed[j];
                ed[j]=-1;
            }
        }
        return ans;
    }
    ACAutomaton():cnt(0),fail(1),ed(1),tree(1){}
};
```

## Manacher

求回文串长度

(s的下标+1)*2对应p数组

p数组的值-1对应了回文串长度

```cpp
vector<int> manacher(string s){
    string cur="^";
    for(char &c:s){
        cur+='#';
        cur+=c;
    }
    cur+='#';
    cur+='@';
    vector<int> p(cur.size(),0);
    int r=0,mid=0;
    for(int i=1;i<cur.size()-1;i++){
        p[i]=i<=r?min(p[2*mid-i],r-i+1):1;
        while(i-p[i]>0&&i+p[i]<cur.size()-1&&cur[i-p[i]]==cur[i+p[i]]) p[i]++;
        if(i+p[i]-1>r){
            r=i+p[i]-1;
            mid=i;
        }
    }
    return p;
}
```

## Trie Tree

```cpp
struct TRIE{
    int tot=0,sz=0;
    vector<vector<int>> tree;
    vector<int> ed;
    TRIE(vector<string> &v,int sz){
        this->sz=sz;
        tree.push_back(vector<int>(sz,0));
        ed.push_back(0);
        for(string &p:v) insert(p);
    }
    void insert(string s){
        int now=0;
        for(char &c:s){
            if(tree[now][ma[c]]==0){
                tree[now][ma[c]]=++tot;
                tree.push_back(vector<int>(sz,0));
                ed.push_back(0);
            }
            now=tree[now][ma[c]];
        }
        ed[now]++;
    }
    int query(string s){
        int now=0,ans=0;
        for(char &c:s){
            if(tree[now][ma[c]]==0) return 0;
            ans=ed[tree[now][ma[c]]];
            now=tree[now][ma[c]];
        }
        return ans;
    }
};
```

## 扩展KMP

对于一个长度为n的字符串，定义函数z[i]，表示s和s[i,n-1] (即以s[i]开头的后缀)的最长公共前缀（LCP）的长度，则z被称为s的z函数，其中z[0]=0。

```cpp
vector<int> z_function(string s){
    int n=(int)s.size();
    vector<int> z(n);
    for(int i=1,l=0,r=0;i<n;i++){
        if(i<=r&&z[i-l]<r-i+1){
            z[i]=z[i-l];
        }else{
            z[i]=max(0ll,r-i+1);
            while(i+z[i]<n&&s[z[i]]==s[i+z[i]]) ++z[i];
        }
        if(i+z[i]-1>r) l=i,r=i+z[i]-1;
    }
    return z;
}
```

## 本质不同的字串数

给定一个长度为n的字符串s，计算s本质不同的子串的数量。

每次在后面新增一个字符c，计算新增的本质不同的子串的数量（以c结尾且之前未出现过的子串）。令t为s+c的反串（将原来的字符串倒序排列），计算出t的$z_{max}$，新增本质不同的子串的数量即为$|t|-z_{max}$

## 前缀函数

$π[i]$ 表示子串 $[0,i]$ 最长的相等的真前缀与真后缀的长度

其中 $π[0]=0$

```cpp
vector<int> prefix_function(string s){
    int n=(int)s.length();
    vector<int> pi(n);
    pi[0]=0;
    for(int i=1;i<n;i++){
        int j=pi[i-1];
        while(j>0&&s[i]!=s[j]) j=pi[j-1];
        if(s[i]==s[j]) j++;
        pi[i]=j;
    }
    return pi;
}
```

## KMP函数

给定一个文本text和一个字符串pattern，找到并展示s在t中的所有出现位置，时间复杂度O(n+m)

```cpp
vector<int> kmp(string text,string pattern){
    string cur=pattern+'#'+text;
    int sz1=text.size(),sz2=pattern.size();
    vector<int> v;
    vector<int> lps=prefix_function(cur);
    for(int i=sz2+1;i<=sz1+sz2;i++){
        if(lps[i]==sz2) v.push_back(i-2*sz2);
    }
    return v;
}
```

## 字符串哈希

```cpp
const int HASHMOD[2]={998244353,(int)1e9+7};
const int BASE[2]={29,31};
struct Stringhash{
    static vector<int> qpow[2];
    vector<int> hash[2];
    void init(){
        qpow[0].push_back(1);
        qpow[1].push_back(1);
        for(int i=1;i<=1e6;i++){
            for(int j=0;j<2;j++){
                qpow[j].push_back(qpow[j].back()*BASE[j]%HASHMOD[j]);
            }
        }
    }
    Stringhash(string s,int base){
        for(int i=0;i<2;i++){
            hash[i]=vector<int>(s.size()+1);
            hash[i][0]=0;
        }
        if(qpow[0].empty()) init();
        for(int i=1;i<=s.size();i++){
            for(int j=0;j<2;j++){
                hash[j][i]=(hash[j][i-1]*BASE[j]%HASHMOD[j]+s[i-1]-base)%HASHMOD[j];
            }
        }
    }
    pair<int,int> gethash(int x,int y){
        pair<int,int> result={0,0};
        for(int i=0;i<2;i++){
            int k=((hash[i][y]-hash[i][x-1]*qpow[i][y-x+1])%HASHMOD[i]+HASHMOD[i])%HASHMOD[i];
            if(i==0) result.first=k;
            else result.second=k;
        }
        return result;
    }
};
vector<int> Stringhash::qpow[2];
```

## 字符串的周期

如果 $s$ 长度为 $r$ 的前缀和长度为 $r$ 的后缀相等，那么 $s$ 长度为 $r$ 的前缀是 $s$ 的一个border，$n-r$ 是 $s$ 的一个周期。

$pi[n-1],pi[pi[n-1]]\dots$ 为所有border的长度

最小周期为 $n-pi[n-1]$

## 最小表示法

$s[i...n]+s[1...i-1]=T$

$s$ 的最小表示法就是与 $s$ 循环同构的字典序最小的字符串

```cpp
int k=0,i=0,j=1;
while(k<n&&i<n&&j<n){
    if(v[(i+k)%n]==v[(j+k)%n]){
        k++;
    }else{
        if(v[(i+k)%n]>v[(j+k)%n]){
            i+=k+1;
        }else{
            j+=k+1;
        }
        if(i==j) i++;
        k=0;
    }
}
i=min(i,j);
```

## 后缀数组（SA）

$sa[i]$ 表示所有后缀排序后第i小的后缀的编号，$rk[i]$ 表示后缀i的排名，$height[i]$ 表示第 $i$ 名后缀与前一名的后缀的最长公共前缀

$height[rk[i]] \ge height[rk[i - 1]] - 1$

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    string s;
    cin>>s;
    auto getsa=[&](const string &s){
        int n=s.size();
        vector<int> sa(n),rk(n);
        iota(sa.begin(),sa.end(),0ll);
        sort(sa.begin(),sa.end(),[&](int a,int b){
            return s[a]<s[b];
        });
        rk[sa[0]]=0;
        for(int i=1;i<n;i++){
            rk[sa[i]]=rk[sa[i-1]];
            if(s[sa[i]]!=s[sa[i-1]]) rk[sa[i]]++;
        }
        for(int len=1;len<n;len<<=1){
            vector<tuple<int,int,int>> v;//第一关键字，第二关键字，下标
            v.reserve(n);
            for(int i=0;i<n;i++){
                v.emplace_back(rk[i],(i+len<n?rk[i+len]:-1),i);
            }
            sort(v.begin(),v.end(),[](const auto &a,const auto &b){
                if(get<0>(a)!=get<0>(b)) return get<0>(a)<get<0>(b);
                if(get<1>(a)!=get<1>(b)) return get<1>(a)<get<1>(b);
                return get<2>(a)<get<2>(b);
            });
            for(int i=0;i<n;i++){
                sa[i]=get<2>(v[i]);
            }
            rk[sa[0]]=0;
            for(int i=1;i<n;i++){
                rk[sa[i]]=rk[sa[i-1]];
                if(get<0>(v[i-1])!=get<0>(v[i])||get<1>(v[i-1])!=get<1>(v[i])){
                    rk[sa[i]]++;
                }
            }
        }
        vector<int> height(n);
        for(int i=0,k=0;i<n;i++){
            if(!rk[i]){
                k=0;
                continue;
            }
            if(k) --k;
            while(i+k<n&&sa[rk[i]-1]+k<n&&s[i+k]==s[sa[rk[i]-1]+k]) k++;
            height[rk[i]]=k;
        }
        return make_pair(sa,height);
    };
    auto [sa,height]=getsa(s);
    for(int i=0;i<s.size();i++) cout<<sa[i]<<" ";
    cout<<"\n"; 
    for(int i=0;i<s.size();i++) cout<<height[i]<<" ";
    cout<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## SAM

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
struct SAM{
    struct edge{
        int len,link,cnt;
        array<int,26> nxt;
        edge(int len):len(len),link(0),cnt(0){nxt.fill(0);}
    };
    vector<edge> node;
    int last;
    SAM():last(1),node(2,0){}
    void add(char c){
        int cur=node.size();
        node.emplace_back(node[last].len+1);
        node[cur].cnt=1;
        int p=last;
        while(p!=0&&!node[p].nxt[c-'a']){
            node[p].nxt[c-'a']=cur;
            p=node[p].link;
        }
        if(p==0){
            node[cur].link=1;
        }else{
            int q=node[p].nxt[c-'a'];
            if(node[q].len==node[p].len+1){
                node[cur].link=q;
            }else{
                int clone=node.size();
                node.emplace_back(node[q]);
                node[clone].cnt=0;
                node[clone].len=node[p].len+1;
                node[q].link=node[cur].link=clone;
                while(p!=0&&node[p].nxt[c-'a']==q){
                    node[p].nxt[c-'a']=clone;
                    p=node[p].link;
                }
            }
        }
        last=cur;
    }
};
void solve(){
    string s;
    cin>>s;
    SAM sam;
    for(char &c:s) sam.add(c);
    vector<vector<int>> son(sam.node.size());
    vector<ll> dp(sam.node.size());
    ll ans=0;
    for(int i=2;i<sam.node.size();i++){
        son[sam.node[i].link].push_back(i);
    }
    vector<int> id(sam.node.size()-1);
    iota(id.begin(),id.end(),1);
    sort(id.begin(),id.end(),[&](int &a,int &b){
        return sam.node[a].len>sam.node[b].len;
    });
    for(auto &x:id){
        dp[x]=sam.node[x].cnt;
        for(auto &p:son[x]){
            dp[x]+=dp[p];
        }
        if(dp[x]>1){
            ans=max(ans,(ll)dp[x]*sam.node[x].len);
        }
    }
    cout<<ans<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

# 数学

## 高精度

压位高精度。在int下，加法压9位，乘法压3位，long long压4位

```cpp
//压位高精，压Base位
template<int Base>
struct BigNum{
    constexpr int pow(int x,int y){
        int ans=1,base=x;
        while(y){
            if(y&1) ans=ans*base;
            base*=base;
            y>>=1;
        }
        return ans;
    }
    const int mod;
    vector<int> v;
    BigNum(int n):mod(pow(10,Base)){
        if(n==0) v.push_back(0);
        while(n){
            v.push_back(n%mod);
            n/=mod;
        }
    }
    vector<int> stos(string &s) const{
        vector<int> v;
        int len=s.size();
        for(int i=len-Base;i+Base-1>=0;i-=Base){
            string k=s.substr(max(0ll,i),min(i+Base-1-max(0ll,i)+1,Base));
            v.push_back(stoi(k));
        }
        return v;
    }
    BigNum(string &s):mod(pow(10,Base)),v(stos(s)){}
    BigNum(vector<int> &_):v(_),mod(pow(10,Base)){};
    BigNum():mod(pow(10,Base)){v.push_back(0);}
    BigNum operator+(const BigNum &e) const{
        vector<int> ans;
        int i;
        for(i=0;i<min(e.v.size(),v.size());i++){
            ans.push_back(e.v[i]+v[i]);
        }
        for(;i<e.v.size();i++){
            ans.push_back(e.v[i]);
        }
        for(;i<v.size();i++){
            ans.push_back(v[i]);
        }
        for(int i=0;i<ans.size();i++){
            if(ans[i]>=mod){
                if(i+1==ans.size()) ans.push_back(0);
                ans[i+1]+=ans[i]/mod;
                ans[i]%=mod;
            }
        }
        while(ans.back()==0) ans.pop_back();
        return BigNum(ans);
    }
    BigNum operator-(const BigNum &e) const{
        vector<int> ans;
        int i;
        for(i=0;i<min(e.v.size(),v.size());i++){
            ans.push_back(v[i]-e.v[i]);
        }
        for(;i<v.size();i++){
            ans.push_back(v[i]);
        }
        for(int i=0;i<ans.size();i++){
            if(ans[i]<0){
                int t=(-ans[i]+mod-1)/mod;
                ans[i]+=t*mod;
                ans[i+1]-=t;
            }
        }
        while(ans.back()==0) ans.pop_back();
        return BigNum(ans);
    }
    BigNum operator*(const BigNum &e) const{
        vector<int> ans;
        for(int i=0;i<v.size();i++){
            for(int j=0;j<e.v.size();j++){
                while(i+j==ans.size()) ans.push_back(0);
                ans[i+j]+=v[i]*e.v[j];
            }
        }
        for(int i=0;i<ans.size();i++){
            if(ans[i]>=mod){
                if(i+1==ans.size()) ans.push_back(0);
                ans[i+1]+=ans[i]/mod;
                ans[i]%=mod;
            }
        }
        while(ans.back()==0) ans.pop_back();
        return BigNum(ans);
    }
    void operator+=(const BigNum &e){
        v=(*this+e).v;
    }
    void operator-=(const BigNum &e){
        v=(*this-e).v;
    }
    void operator*=(const BigNum &e){
        v=(*this*e).v;
    }
    void operator=(int x){
        v.clear();
        if(x==0) v.push_back(0);
        else{
            while(x){
                v.push_back(x%mod);
                x/=mod;
            }
        }
    }
    void operator=(const BigNum &e){
        v=e.v;
    }
    BigNum operator+(int x) const{
        return (*this)+BigNum<Base>(x);
    }
    BigNum operator*(int x) const{
        return (*this)*BigNum<Base>(x);
    }
    int getlen() const{
        if(v.empty()) return 0;
        int len=0;
        if(v.size()==1){
            if(v.front()==0) return 1;
            int tmp=v.front();
            while(tmp){
                len++;
                tmp/=10;
            }
            return len;
        }else{
            int tmp=v.back();
            while(tmp){
                len++;
                tmp/=10;
            }
            return len+Base*(v.size()-1);
        }
    }
    bool operator<(const BigNum &e) const{
        int len1=getlen(),len2=e.getlen();
        if(len1!=len2) return len1<len2;
        for(int i=v.size()-1;i>=0;i--){
            if(v[i]<e.v[i]) return 1;
            if(v[i]>e.v[i]) return 0;
        }
        return 0;
    }
    friend ostream& operator<<(ostream& os,const BigNum& obj){
        for(int i=obj.v.size()-1;i>=0;i--){
            if(i!=obj.v.size()-1){
                for(int j=obj.mod/10;j>=1;j/=10){
                    os<<obj.v[i]/j%10;
                }
            }else{
                os<<obj.v[i];
            }
        }
        return os;
    }
    friend istream& operator>>(istream& is,BigNum& obj){
        string s;
        is>>s;
        obj.v=obj.stos(s);
        return is;
    }
};
```

```cpp
const int base = 1000;
const int base_digits = 3;  // 分解为九个数位一个数字
struct bigint {
    vector<int> a;
    int sign;

    bigint() : sign(1) {}
    bigint operator-() const {
        bigint res = *this;
        res.sign = -sign;
        return res;
    }
    bigint(long long v) {
        *this = v;
    }
    bigint(const string &s) {
        read(s);
    }
    void operator=(const bigint &v) {
        sign = v.sign;
        a = v.a;
    }
    void operator=(long long v) {
        a.clear();
        sign = 1;
        if (v < 0)
            sign = -1, v = -v;
        for (; v > 0; v = v / base) {
            a.push_back(v % base);
        }
    }

    // 基础加减乘除
    bigint operator+(const bigint &v) const {
        if (sign == v.sign) {
            bigint res = v;
            for (int i = 0, carry = 0; i < (int)max(a.size(), v.a.size()) || carry; ++i) {
                if (i == (int)res.a.size()) {
                    res.a.push_back(0);
                }
                res.a[i] += carry + (i < (int)a.size() ? a[i] : 0);
                carry = res.a[i] >= base;
                if (carry) {
                    res.a[i] -= base;
                }
            }
            return res;
        }
        return *this - (-v);
    }
    bigint operator-(const bigint &v) const {
        if (sign == v.sign) {
            if (abs() >= v.abs()) {
                bigint res = *this;
                for (int i = 0, carry = 0; i < (int)v.a.size() || carry; ++i) {
                    res.a[i] -= carry + (i < (int)v.a.size() ? v.a[i] : 0);
                    carry = res.a[i] < 0;
                    if (carry) {
                        res.a[i] += base;
                    }
                }
                res.trim();
                return res;
            }
            return -(v - *this);
        }
        return *this + (-v);
    }
    void operator*=(int v) {
        check(v);
        for (int i = 0, carry = 0; i < (int)a.size() || carry; ++i) {
            if (i == (int)a.size()) {
                a.push_back(0);
            }
            long long cur = a[i] * (long long)v + carry;
            carry = (int)(cur / base);
            a[i] = (int)(cur % base);
        }
        trim();
    }
    void operator/=(int v) {
        check(v);
        for (int i = (int)a.size() - 1, rem = 0; i >= 0; --i) {
            long long cur = a[i] + rem * (long long)base;
            a[i] = (int)(cur / v);
            rem = (int)(cur % v);
        }
        trim();
    }
    int operator%(int v) const {
        if (v < 0) {
            v = -v;
        }
        int m = 0;
        for (int i = a.size() - 1; i >= 0; --i) {
            m = (a[i] + m * (long long)base) % v;
        }
        return m * sign;
    }

    void operator+=(const bigint &v) {
        *this = *this + v;
    }
    void operator-=(const bigint &v) {
        *this = *this - v;
    }
    bigint operator*(int v) const {
        bigint res = *this;
        res *= v;
        return res;
    }
    bigint operator/(int v) const {
        bigint res = *this;
        res /= v;
        return res;
    }
    void operator%=(const int &v) {
        *this = *this % v;
    }

    bool operator<(const bigint &v) const {
        if (sign != v.sign)
            return sign < v.sign;
        if (a.size() != v.a.size())
            return a.size() * sign < v.a.size() * v.sign;
        for (int i = a.size() - 1; i >= 0; i--)
            if (a[i] != v.a[i])
                return a[i] * sign < v.a[i] * sign;
        return false;
    }
    bool operator>(const bigint &v) const {
        return v < *this;
    }
    bool operator<=(const bigint &v) const {
        return !(v < *this);
    }
    bool operator>=(const bigint &v) const {
        return !(*this < v);
    }
    bool operator==(const bigint &v) const {
        return !(*this < v) && !(v < *this);
    }
    bool operator!=(const bigint &v) const {
        return *this < v || v < *this;
    }

    bigint abs() const {
        bigint res = *this;
        res.sign *= res.sign;
        return res;
    }
    void check(int v) {  // 检查输入的是否为负数
        if (v < 0) {
            sign = -sign;
            v = -v;
        }
    }
    void trim() {  // 去除前导零
        while (!a.empty() && !a.back()) a.pop_back();
        if (a.empty())
            sign = 1;
    }
    bool isZero() const {  // 判断是否等于零
        return a.empty() || (a.size() == 1 && !a[0]);
    }
    friend bigint gcd(const bigint &a, const bigint &b) {
        return b.isZero() ? a : gcd(b, a % b);
    }
    friend bigint lcm(const bigint &a, const bigint &b) {
        return a / gcd(a, b) * b;
    }
    void read(const string &s) {
        sign = 1;
        a.clear();
        int pos = 0;
        while (pos < (int)s.size() && (s[pos] == '-' || s[pos] == '+')) {
            if (s[pos] == '-')
                sign = -sign;
            ++pos;
        }
        for (int i = s.size() - 1; i >= pos; i -= base_digits) {
            int x = 0;
            for (int j = max(pos, i - base_digits + 1); j <= i; j++) x = x * 10 + s[j] - '0';
            a.push_back(x);
        }
        trim();
    }
    friend istream &operator>>(istream &stream, bigint &v) {
        string s;
        stream >> s;
        v.read(s);
        return stream;
    }
    friend ostream &operator<<(ostream &stream, const bigint &v) {
        if (v.sign == -1)
            stream << '-';
        stream << (v.a.empty() ? 0 : v.a.back());
        for (int i = (int)v.a.size() - 2; i >= 0; --i)
            stream << setw(base_digits) << setfill('0') << v.a[i];
        return stream;
    }

    /* 大整数乘除大整数部分 */
    typedef vector<long long> vll;
    bigint operator*(const bigint &v) const {  // 大整数乘大整数
        vector<int> a6 = convert_base(this->a, base_digits, 6);
        vector<int> b6 = convert_base(v.a, base_digits, 6);
        vll a(a6.begin(), a6.end());
        vll b(b6.begin(), b6.end());
        while (a.size() < b.size()) a.push_back(0);
        while (b.size() < a.size()) b.push_back(0);
        while (a.size() & (a.size() - 1)) a.push_back(0), b.push_back(0);
        vll c = karatsubaMultiply(a, b);
        bigint res;
        res.sign = sign * v.sign;
        for (int i = 0, carry = 0; i < (int)c.size(); i++) {
            long long cur = c[i] + carry;
            res.a.push_back((int)(cur % 1000000));
            carry = (int)(cur / 1000000);
        }
        res.a = convert_base(res.a, 6, base_digits);
        res.trim();
        return res;
    }
    friend pair<bigint, bigint> divmod(const bigint &a1,
                                       const bigint &b1) {  // 大整数除大整数，同时返回答案与余数
        int norm = base / (b1.a.back() + 1);
        bigint a = a1.abs() * norm;
        bigint b = b1.abs() * norm;
        bigint q, r;
        q.a.resize(a.a.size());
        for (int i = a.a.size() - 1; i >= 0; i--) {
            r *= base;
            r += a.a[i];
            int s1 = r.a.size() <= b.a.size() ? 0 : r.a[b.a.size()];
            int s2 = r.a.size() <= b.a.size() - 1 ? 0 : r.a[b.a.size() - 1];
            int d = ((long long)base * s1 + s2) / b.a.back();
            r -= b * d;
            while (r < 0) r += b, --d;
            q.a[i] = d;
        }
        q.sign = a1.sign * b1.sign;
        r.sign = a1.sign;
        q.trim();
        r.trim();
        return make_pair(q, r / norm);
    }
    static vector<int> convert_base(const vector<int> &a, int old_digits, int new_digits) {
        vector<long long> p(max(old_digits, new_digits) + 1);
        p[0] = 1;
        for (int i = 1; i < (int)p.size(); i++) p[i] = p[i - 1] * 10;
        vector<int> res;
        long long cur = 0;
        int cur_digits = 0;
        for (int i = 0; i < (int)a.size(); i++) {
            cur += a[i] * p[cur_digits];
            cur_digits += old_digits;
            while (cur_digits >= new_digits) {
                res.push_back((int)(cur % p[new_digits]));
                cur /= p[new_digits];
                cur_digits -= new_digits;
            }
        }
        res.push_back((int)cur);
        while (!res.empty() && !res.back()) res.pop_back();
        return res;
    }
    static vll karatsubaMultiply(const vll &a, const vll &b) {
        int n = a.size();
        vll res(n + n);
        if (n <= 32) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    res[i + j] += a[i] * b[j];
                }
            }
            return res;
        }

        int k = n >> 1;
        vll a1(a.begin(), a.begin() + k);
        vll a2(a.begin() + k, a.end());
        vll b1(b.begin(), b.begin() + k);
        vll b2(b.begin() + k, b.end());

        vll a1b1 = karatsubaMultiply(a1, b1);
        vll a2b2 = karatsubaMultiply(a2, b2);

        for (int i = 0; i < k; i++) a2[i] += a1[i];
        for (int i = 0; i < k; i++) b2[i] += b1[i];

        vll r = karatsubaMultiply(a2, b2);
        for (int i = 0; i < (int)a1b1.size(); i++) r[i] -= a1b1[i];
        for (int i = 0; i < (int)a2b2.size(); i++) r[i] -= a2b2[i];

        for (int i = 0; i < (int)r.size(); i++) res[i + k] += r[i];
        for (int i = 0; i < (int)a1b1.size(); i++) res[i] += a1b1[i];
        for (int i = 0; i < (int)a2b2.size(); i++) res[i + n] += a2b2[i];
        return res;
    }

    void operator*=(const bigint &v) {
        *this = *this * v;
    }
    bigint operator/(const bigint &v) const {
        return divmod(*this, v).first;
    }
    void operator/=(const bigint &v) {
        *this = *this / v;
    }
    bigint operator%(const bigint &v) const {
        return divmod(*this, v).second;
    }
    void operator%=(const bigint &v) {
        *this = *this % v;
    }
};
```

## 矩阵相关

```cpp
const double eps=1e-8;
struct matrix{
    vector<vector<double>> v;
    int n,m;
    matrix(int n,int m):n(n),m(m){
        v.assign(n,vector<double>(m));
    }
    matrix(vector<vector<double>> &v):v(v),n(v.size()),m(v[0].size()){}
    //第n行第m列变成x
    void set(int n,int m,double x){
        v[n-1][m-1]=x;
    }
    matrix operator*(const matrix &e) const{
        matrix ans(n,e.m);
        for(int i=0;i<n;i++){
            for(int k=0;k<m;k++){
                for(int j=0;j<e.m;j++){
                    ans.v[i][j]+=v[i][k]*e.v[k][j];
                }
            }
        }
        return ans;
    }
    //高斯消元
    //无解-1，无穷解0，有唯一解1
    int Gauss(){
        int column=0;
        for(int i=0;i<n;i++){
            while(column<m){
                int line=i;
                double maxn=v[i][column];
                for(int j=i+1;j<n;j++){
                    if(fabs(v[j][column])>abs(maxn)){
                        maxn=v[j][column];
                        line=j;
                    }
                }
                swap(v[i],v[line]);
                if(fabs(v[i][column])<eps){
                    column++;
                    continue;
                }
                double k=v[i][column];
                for(int j=column;j<m;j++){
                    v[i][j]/=k;
                }
                for(int j=0;j<n;j++){
                    if(j==i) continue;
                    k=v[j][column];
                    for(int z=column;z<m;z++){
                        v[j][z]-=k*v[i][z];
                    }
                }
                break;
            }
        }
        int inf=0;
        for(int i=0;i<n;i++){
            bool ok=0;
            for(int j=0;j<m-1;j++){
                if(fabs(v[i][j])>eps){
                    ok=1;
                    break;
                }
            }
            if(!ok){
                if(fabs(v[i][m-1])>eps) return -1;
                inf++;
            }
        }
        return inf==0;
    }
};
```

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
int quickpow(int x,int y,int mod){
    int ans=1,base=x;
    while(y){
        if(y&1) ans=ans*base%mod;
        base=base*base%mod;
        y>>=1;
    }
    return ans;
}
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> v(n,vector<int>(n+1));
    for(int i=0;i<n;i++){
        int k;
        cin>>k;
        v[i][i]=1;
        while(k--){
            int x;
            cin>>x;
            v[x-1][i]=1;
        }
    }
    vector<int> s(n),t(n);
    for(int i=0;i<n;i++){
        cin>>s[i];
    }
    for(int i=0;i<n;i++){
        cin>>t[i];
    }
    for(int i=0;i<n;i++){
        v[i][n]=((t[i]-s[i])%m+m)%m;
    }
    vector<pair<int,int>> zy;
    auto gauss=[&](int N,int M){
        int cur=0;
        for(int i=0;i<M;i++){
            int now=cur;
            while(now<n&&!v[now][i]) now++;
            if(now==n) continue;
            if(now!=cur) swap(v[now],v[cur]);
            zy.emplace_back(cur,i);
            for(int j=0;j<n;j++){
                if(j==cur||!v[j][i]) continue;
                const int inv=quickpow(v[cur][i],m-2,m)*v[j][i]%m;
                for(int k=i;k<M;k++){
                    v[j][k]=((v[j][k]-inv*v[cur][k])%m+m)%m;
                }
            }
            cur++;
        }
    };
    gauss(n,n+1);
    if(zy.size()&&zy.back().second==n){
        cout<<"niuza\n";
        return;
    }
    vector<int> ans(n);
    int pre=n;
    auto combine=[&](int x,int t){
        for(int i=0;i<n;i++){
            v[i][n]=((v[i][n]-t*v[i][x])%m+m)%m;
        }
    };
    for(int i=(int)(zy.size())-1;i>=0;i--){
        const int inv=quickpow(v[zy[i].first][zy[i].second],m-2,m)*v[zy[i].first][n]%m;
        ans[zy[i].second]=inv;
        combine(zy[i].second,inv);
    }
    for(int &p:ans) cout<<p<<" ";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

高斯消元解异或方程组，所有的加减乘除操作变成异或

```cpp
//异或版本
struct matrix{
    vector<vector<int>> v;
    int n,m;
    matrix(int n,int m):n(n),m(m){
        v.assign(n,vector<int>(m));
    }
    matrix(vector<vector<int>> &v):v(v),n(v.size()),m(v[0].size()){}
    //第n行第m列变成x
    void set(int n,int m,int x){
        v[n-1][m-1]=x;
    }
    matrix operator*(const matrix &e) const{
        vector<vector<int>> ans(n,vector<int>(e.m,0));
        for(int i=0;i<n;i++){
            for(int j=0;j<e.m;j++){
                for(int k=0;k<m;k++){
                    ans[i][j]+=v[i][k]*e.v[k][j];
                }
            }
        }
        return matrix(ans);
    };
    //高斯消元
    //无解-1，无穷解0，有唯一解1
    int Gauss(){
        int column=0;
        for(int i=0;i<n;i++){
            while(column<m){
                int line=i;
                int maxn=v[i][column];
                for(int j=i+1;j<n;j++){
                    if(abs(v[j][column])>maxn){
                        maxn=v[j][column];
                        line=j;
                    }
                }
                swap(v[i],v[line]);
                if(v[i][column]==0){
                    column++;
                    continue;
                }
                for(int j=0;j<n;j++){
                    if(j==i) continue;
                    int k=v[j][column];
                    if(k){
                        for(int z=column;z<m;z++){
                            v[j][z]^=v[i][z];
                        }
                    }
                }
                break;
            }
        }
        int inf=0;
        for(int i=0;i<n;i++){
            bool ok=0;
            for(int j=0;j<m-1;j++){
                if(v[i][j]!=0){
                    ok=1;
                    break;
                }
            }
            if(!ok){
                if(v[i][m-1]!=0) return -1;
                inf++;
            }
        }
        return inf==0;
    }
};
```

```cpp
#include<bits/stdc++.h>
using namespace std;
void solve(){
    int n,k;
    cin>>n>>k;
    vector<vector<int>> v(k+1,vector<int>(n));
    vector<pair<int,int>> zy;
    auto gauss=[&](int n,int m){
        int cur=0;
        for(int i=0;i<m;i++){
            int now=cur;
            while(now<n&&!v[now][i]) now++;
            if(now==n) continue;
            if(now!=cur) swap(v[now],v[cur]);
            zy.emplace_back(cur,i);
            for(int j=0;j<n;j++){
                if(cur!=j&&v[j][i]){
                    for(int k=0;k<m;k++){
                        v[j][k]^=v[cur][k];
                    }
                }
            }
            cur++;
        }
        return;
    };
    for(int i=0;i<n;i++){
        string s;
        cin>>s;
        for(int j=0;j<k;j++){
            v[j][i]=s[j]-'0';
        }
        v[k][i]=1;
    }
    gauss(k+1,n);
    if(zy.size()==n){
        cout<<"*\n";
    }else{
        vector<int> pre(k+1),ans;
        int now=zy.size()-1;
        auto combine=[&](int x){
            for(int i=0;i<=k;i++){
                pre[i]^=v[i][x];
            }
        };
        for(int i=n-1;i>=0;i--){
            //如果是主元列
            if(i==zy[now].second){
                if(pre[zy[now].first]==1){
                    ans.push_back(i);
                    combine(i);
                }
                now--;
            }else{
                ans.push_back(i);
                combine(i);
            }
        }
        assert(count(pre.begin(),pre.end(),1)==0);
        vector<int> result(n);
        for(int i=0;i<ans.size();i++){
            if(i<ans.size()/2) result[ans[i]]=1;
            else result[ans[i]]=2;
        }
        for(int &p:result) cout<<p;
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 卡特兰数

    有一个大小为 $n×n$ 的方格图，左下角为 $(0,0)$ ，右上角为 $(n,n)$ ，从左下角开始每次只能向右或者向上走一个单位，不能走到 $y=x$ 上方（但可以触碰），有几种可能的路径

前几项：$1\ 1\ 2\ 5\ 14\ 42$

递推式：
$$
H_n=\frac{\binom{2n}{n}}{n+1}(n\ge2,n\in N_+)
$$

$$
H_n=\begin{cases}
\sum_1^n H_{i-1}H_{n-i},n\ge2,&n\in N_+\\
1,&n=0,1
\end{cases}
$$

$$
H_n=\frac{H_{n-1}(4n-2)}{n+1}
$$

$$
H_n=\binom{2n}{n}-\binom{2n}{n-1}
$$

推论：

求 $C(a,b,k)$ ，使得任意前缀的 $cnt_a+k>cnt_b$

即从 $(0,0)$ 走到 $(a,b)$ ，不能走到 $y=x+k-1$ 上方
$$
C(a,b,k)=\begin{cases}
0,&a+k<b\\
\binom{a+b}{a}-\binom{a+b}{a+k},&else
\end{cases}
$$

## MillerRabin

判断某个数是否是质数

```cpp
struct MillerRabin{
    vector<int> Prime;
    MillerRabin():Prime({2,3,5,7,11,13,17,19,23}){}
    static constexpr int mulp(const int &a,const int &b,const int &P){
        int res=a*b-(int)(1.L*a*b/P)*P;
        res%=P;
        res+=(res<0?P:0);
        return res;
    }
    static constexpr int powp(int a,int mi,const int &mod){
        int ans = 1;
        for(;mi;mi>>=1){
            if(mi&1) ans=mulp(ans,a,mod);
            a=mulp(a,a,mod);
        }
        return ans;
    }
    bool operator()(const int &v){
        if(v<2||v!=2&&v%2==0) return false;
        int s=v-1;
        while(!(s&1)) s>>=1;
        for(int x:Prime){
            if(v==x) return true;
            int t=s,m=powp(x,s,v);
            while(t!=v-1&&m!=1&&m!=v-1) m=mulp(m,m,v),t<<=1;
            if(m!=v-1&&!(t&1))return false;
        }
        return true;
    }
};
```

## PollardRho

判断质数（使用millerrabin判断），计算因子

```cpp
struct PollardRho:public MillerRabin{
    mt19937 myrand;
    PollardRho():myrand(time(0)){}
    int rd(int l,int r){
        return myrand()%(r-l+1)+l;
    }
    int operator()(int n) { //返回n的随机一个[2, n-1]内的因子,或者判定是质数
        if(n==4) return 2;
        MillerRabin &super=*this;
        //如果n是质数直接返回n
        if(super(n)) return n; 
        while(1){
            int c=rd(1,n-1);
            auto f=[&](int x){
                return (super.mulp(x,x,n)+c)%n;
            };
            int t=0,r=0,p=1,q;
            do{
                for(int i=0;i<128;i++){
                    t=f(t),r=f(f(r));
                    if(t==r||(q=super.mulp(p,abs(t-r),n))==0) break;
                    p=q;
                }
                int d=__gcd(p,n);
                if(d>1) return d;
            }while(t!=r);
        }
    }
};
```

## 超快质因数分解&求约数

时间复杂度$O(n^{\frac{1}{4}})$

```cpp
stack<int> st;
st.push(x);
map<int,int> ma;
while(!st.empty()){
    int f=st.top();
    st.pop();
    int k=rho(f);
    if(k==f){
        ma[k]++;
    }else{
        st.push(k);
        st.push(f/k);
    }
}
vector<pair<int,int>> v;
for(auto &[p,q]:ma) v.push_back({p,q});
function<void(int,int)> dfs=[&](int id,int now){
    if(id==v.size()){
        if(i<now&&cal(now)==i) ans++;
        return;
    }
    for(int i=0;i<=v[id].second;i++){
        dfs(id+1,now);
        now*=v[id].first;
    }
};
```

## 线性筛质数

```cpp
struct EulerSieve{
    vector<int> prime;
    vector<int> v;
    int n;
    EulerSieve(int n):v(n+1){
        this->n=n;
        for(int i=2;i<=n;i++){
            if(v[i]==0){
                prime.push_back(i);
                v[i]=i;
            }
            for(int &p:prime){
                if(i*p>n) break;
                v[i*p]=p;
                if(i%p==0) break;
            }
        }
    }
    vector<int> getdiv(int x) const{
        vector<int> _div(1,1);
        while(x>1){
            int d=v[x];
            int l=0,r=_div.size();
            while(x%d==0){
                for(int k=l;k<r;k++){
                    _div.push_back(_div[k]*d);
                }
                x/=d;
                l=r;
                r=_div.size();
            }
        }
        return _div;
    }
};
```

## 线性筛欧拉函数

小于等于n且与n互质的正整数数量

```cpp
struct EulerSieve {
    int n;
    vector<int> prime;
    vector<bool> isprime;
    vector<int> phi;

    EulerSieve(int n) : n(n), isprime(n + 1, true), phi(n + 1) {
        isprime[0] = isprime[1] = false;
        phi[0] = 0;
        phi[1] = 1;

        for (int i = 2; i <= n; i++) {
            if (isprime[i]) {
                prime.push_back(i);
                phi[i] = i - 1;
            }
            for (int &p : prime) {
                if (i * p > n) break;
                isprime[i * p] = false;
                if (i % p == 0) {
                    phi[i * p] = phi[i] * p;
                    break;
                } else {
                    phi[i * p] = phi[i] * (p - 1);
                }
            }
        }
    }
};
```

## 直接求欧拉函数

```cpp
int phi(int n){
    int ans=n;
    for(int i=2;i*i<=n;i++){
        if(n%i==0){
            ans=ans/i*(i-1);
            while(n%i==0) n/=i;
        }
    }
    if(n>1) ans=ans/n*(n-1);
    return ans;
}
```

## 线性筛莫比乌斯函数

```cpp
struct EulerSieve{
    int n;
    vector<int> prime;
    vector<bool> isprime;
    vector<int> mu;

    EulerSieve(int n) : n(n), isprime(n + 1, true), mu(n + 1){
        isprime[0] = isprime[1] = false;
        mu[0] = 0;
        mu[1] = 1;
        for (int i = 2; i <= n; i++){
            if (isprime[i]){
                prime.push_back(i);
                mu[i] = -1;
            }
            for (int &p : prime){
                if (i * p > n)
                    break;
                isprime[i * p] = false;
                if (i % p == 0)
                    break;
                mu[i * p] = -mu[i];
            }
        }
    }
};
```

## 莫比乌斯函数

$$
\mu(n) = \begin{cases}
1, & \text{若} n = 1 \\
0, & \text{若} n \text{能被一个素数的平方整除} \\
(-1)^r, & \text{若} n \text{是} r \text{个} (r \geq 1) \text{不同素数的乘积}
\end{cases}
$$

## 线性筛约数个数

g是约数个数，cnt是最小质因子的幂次

```cpp
struct EulerSieve{
    vector<int> prime;
    vector<int> v,g,cnt;
    int n;
    EulerSieve(int n):v(n+1),g(n+1),cnt(n+1){
        this->n=n;
        g[1]=1;
        cnt[1]=0;
        for(int i=2;i<=n;i++){
            if(v[i]==0){
                prime.push_back(i);
                v[i]=i;
                g[i]=2;
                cnt[i]=1;
            }
            for(int &p:prime){
                if(i*p>n) break;
                v[i*p]=p;
                if(i%p==0){
                    g[i*p]=g[i]/(cnt[i]+1)*(cnt[i]+2);
                    cnt[i*p]=cnt[i]+1;
                    break;
                }
                g[i*p]=g[i]*g[p];
                cnt[i*p]=1;
            }
        }
    }
    vector<int> getdiv(int x) const{
        vector<int> _div(1,1);
        while(x>1){
            int d=v[x];
            int l=0,r=_div.size();
            while(x%d==0){
                for(int k=l;k<r;k++){
                    _div.push_back(_div[k]*d);
                }
                x/=d;
                l=r;
                r=_div.size();
            }
        }
        return _div;
    }
};
```

## 线性筛约数和

```cpp
struct EulerSieve{
    vector<int> prime;
    vector<int> v,g,sump;
    //sump 1+p^1+p^2+...+p^k
    int n;
    EulerSieve(int n):v(n+1),g(n+1),sump(n+1){
        this->n=n;
        g[1]=1;
        sump[1]=1;
        for(int i=2;i<=n;i++){
            if(v[i]==0){
                prime.push_back(i);
                v[i]=i;
                g[i]=i+1;
                sump[i]=1+i;
            }
            for(int &p:prime){
                if(i*p>n) break;
                v[i*p]=p;
                if(i%p==0){
                    sump[i*p]=sump[i]*p+1;
                    g[i*p]=g[i]/sump[i]*sump[i*p];
                    break;
                }
                sump[i*p]=p+1;
                g[i*p]=g[i]*g[p];
            }
        }
    }
    vector<int> getdiv(int x) const{
        vector<int> _div(1,1);
        while(x>1){
            int d=v[x];
            int l=0,r=_div.size();
            while(x%d==0){
                for(int k=l;k<r;k++){
                    _div.push_back(_div[k]*d);
                }
                x/=d;
                l=r;
                r=_div.size();
            }
        }
        return _div;
    }
};
```

## 组合数学

```cpp
template<int MOD>
struct Comb{
    vector<int> jc,ijc;
    int quickpow(int x,int y){
        x%=MOD;
        if(x==0) return 0;
        int ans=1,base=x;
        while(y){
            if(y&1) ans=ans*base%MOD;
            base=base*base%MOD;
            y>>=1;
        }
        return ans;
    }
    Comb(int n){
        jc.resize(n+1);
        ijc.resize(n+1);
        jc[0]=1;
        for(int i=1;i<=n;i++) jc[i]=jc[i-1]*i%MOD;
        ijc[n]=quickpow(jc[n],MOD-2);
        for(int i=n-1;i>=0;i--) ijc[i]=ijc[i+1]*(i+1)%MOD;
    }
    int C(int n,int k){
        if(n<0||k<0||n<k) return 0;
        return jc[n]*ijc[k]%MOD*ijc[n-k]%MOD;
    }
    int A(int n,int k){
        if(n<0||k-1<0||n<k) return 0;
        return jc[n]*ijc[n-k]%MOD;
    }
    int CLucas(int n,int m){
        if(m==0) return 1;
        return C(n%MOD,m%MOD)*CLucas(n/MOD,m/MOD)%MOD;
    }
    int Stirling2(int n,int m){
        int ans=0;
        for(int i=0;i<=m;i++){
            ans=(ans+((m-i)%2==0?1:-1)*quickpow(i,n)%MOD*ijc[i]%MOD*ijc[m-i]%MOD)%MOD;
        }
        return ans;
    }
};
```

## 卢卡斯定理

用于求解问题规模很大，而模数是一个不大的质数的时候的组合数问题，p为质数
$$
C^m_n~mod~p=
\begin{cases}
1~~m=0\\
C_{[\frac{n}{p}]}^{[\frac{m}{p}]}*C^{m~mod~p}_{n~mod~p}~mod~p
\end{cases}
$$

$$
C^m_n~mod~p=C^{b_0}_{a_0}\cdot C^{b_1}_{a_1}\cdot C^{b_2}_{a_2} \cdot\cdot\cdot C^{b_k}_{a_k}
$$

$$
n=a_0+a_1p+a_2p^2+a_3p^3
\\
m=b_0+b_1p+b_2p^2+b_3p^3
\\
m、n按p进制展开
$$

n&m==m,$C_n^m$为奇数

```cpp
// template <int P>
struct Lucas {
    int P;  // prime mod
    vector<i64> fac, ifac;

    Lucas(int p) : P(p) {
        fac.resize(p);
        ifac.resize(p);
        fac[0] = ifac[0] = 1;
        for (int i = 1; i < p; i++) {
            fac[i] = fac[i - 1] * i % P;
        }
        ifac[p - 1] = qpow(fac[p - 1], P - 2, P);
        for (int i = p - 1; i >= 1; i--) {
            ifac[i - 1] = ifac[i] * i % P;
        }
    }

    // C(n,m) % P: only for 0 <= n,m < P
    i64 C(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac[n] * ifac[m] % P * ifac[n - m] % P;
    }

    // C(n,m) % P: Lucas theorem
    i64 lucas(i64 n, i64 m) {
        if (m == 0) return 1;
        return C(n % P, m % P) * lucas(n / P, m / P) % P;
    }
};
// Lucas comb(p); comb.lucas(n,m) => C(n,m) % p
```

### exLucas

对于 $\mathrm{mod}$ 不是素数的情况（素数幂 or 一般合数），质因数分解得 $\mathrm{mod}=p_1^{\alpha_1}p_2^{\alpha_2}\cdots p_k^{\alpha_k}$，分别求出模 $p_i ^{\alpha_i}$ 意义下组合数 $C(n,m)$ 的余数，对同余方程组用 $\text{CRT}$ 合并答案。

- 时间复杂度：$O(\sqrt{\mathrm{mod}}+\sum{p_i^{\alpha_i}})$
- 如果 $\mathrm{mod}$ 可以很容易分解为**幂次为1的质数**，建议用Lucas对每个 $p_i$ 计算后CRT合并，而不是使用exLucas

```cpp
// C(n,m) % mod, mod can be composite
class ExLucas {
public:
    static i64 exlucas(i64 n, i64 m, i64 mod) {
        if (mod == 1 || n < m || m < 0) return 0;
        vector<i64> mods, a;
        i64 x = mod;
        for (i64 i = 2; i * i <= x; i++) {
            if (x % i == 0) {
                i64 pk = 1;
                while (x % i == 0) {
                    x /= i;
                    pk *= i;
                }
                mods.push_back(pk);
                a.push_back(C(n, m, i, pk));
            }
        }
        if (x > 1) {
            mods.push_back(x);
            a.push_back(C(n, m, x, x));
        }
        return CRT(a, mods);
    }

private:
    static i64 fac(i64 n, i64 p, i64 pk, const vector<i64> &pre) {
        if (n == 0) return 1;
        i64 res = pre[pk];
        res = qpow(res, n / pk, pk);
        res = res * pre[n % pk] % pk;
        return res * fac(n / p, p, pk, pre) % pk;
    }
    static i64 inv(i64 a, i64 m) {
        i64 x, y;
        exgcd(a, m, x, y);
        return (x % m + m) % m;
    }
    // C(n,m) % pk, p is prime factor of pk
    static i64 C(i64 n, i64 m, i64 p, i64 pk) {
        if (n < m || m < 0) return 0;
        // O(pk) 预处理前缀积
        vector<i64> pre(pk + 1);
        pre[0] = 1;
        for (i64 i = 1; i <= pk; i++) {
            if (i % p == 0) {
                pre[i] = pre[i - 1];
            } else {
                pre[i] = pre[i - 1] * i % pk;
            }
        }
        i64 f1 = fac(n, p, pk, pre);
        i64 f2 = fac(m, p, pk, pre);
        i64 f3 = fac(n - m, p, pk, pre);
        i64 res = 0;
        for (i64 i = n; i; i /= p) res += i / p;
        for (i64 i = m; i; i /= p) res -= i / p;
        for (i64 i = n - m; i; i /= p) res -= i / p;
        return f1 * inv(f2, pk) % pk * inv(f3, pk) % pk * qpow(p, res, pk) % pk;
    }
};
// Exlucas::exlucas(n, m, mod) => C(n,m) % mod
```

## 上指标求和

$$
\Sigma_{i=0}^n{i\choose k}={n+1\choose k+1}
$$

考虑添加一个虚拟的第$k+1$个元素，枚举这个元素所占据的位置为$i$，则前$k$个元素应在前$i-1$个位置排列，$i$的范围为$[1,n+1]$，则有
$$
{n+1\choose k+1}=\Sigma_{i=1}^{n+1}{i-1\choose k}=\Sigma_{i=0}^n{i\choose k}
$$

## 双阶乘

写作 $n!!$ ，表示不超过这个正整数且与它有相同的奇偶性所有正整数乘积

某些情况下可以与单阶乘互相转化：

$(2n)!!=2\times 4\times 6\times \dots \times 2n=2^nn!$

$(2n-1)!!=1\times 3\times 5\times \dots \times (2n-1)=\frac{(2n)!}{(2n)!!}=\frac{(2n)!}{2^nn!}$

## 二项式反演

$$
\begin{aligned}
g_n &= \sum_{i=0}^n C_n^i f_i \\
f_n &= \sum_{i=0}^n C_n^i (-1)^{n-i} g_i
\end{aligned}
$$

$$
\begin{aligned}
g_k &= \sum_{i=k}^n C_i^k f_i \\
f_k &= \sum_{i=k}^n C_i^k (-1)^{i-k} g_i
\end{aligned}
$$

## 第二类斯特林数

将n个两两不同的元素，划分为k个互不区分的非空子集的方案数

## 康托展开

用于全排列的状态压缩，是一个全排列到一个自然数的映射，康托展开的实质是计算当前排列在所有从小到大的排列中的次序编号

康托展开的表达式为$X=a_n(n-1)!+a_{n-1}(n-2)!+...+a_1\cdot0!$

其中X为比当前排列小的全排列个数，（X+1）即为当前排列的次序编号，n表示全排列的长度，$a_i$表示原排列中的第i位（从右往左从低到高）在当前未出现（剩下未被选择）的元素集合中比其小的元素个数

时间复杂度n^2，用树状数组可优化为nlogn

还原：先让排名-1，从高位开始，每轮整除i！，即可得到当前位有多少个数小于他（去掉已经存在的），线段树优化为logn

```c++
struct Cantor{
    struct SegmentTree{
        vector<int> tree;
        int n;
        SegmentTree(int n):tree((n<<2)+10,0),n(n){}
        void update(int x,int delta){
            function<void(int,int,int,int,int)> realupdate=[&](int id,int l,int r,int x,int delta){
                tree[id]+=delta;
                if(l==r){
                    return;
                }
                int mid=l+(r-l>>1);
                if(x<=mid) realupdate(id<<1,l,mid,x,delta);
                else realupdate(id<<1|1,mid+1,r,x,delta);
            };
            realupdate(1,1,n,x,delta);
        }
        int querysum(int x,int y){
            if(x>y) return 0;
            function<int(int,int,int,int,int)> realquerysum=[&](int id,int l,int r,int x,int y){
                if(x<=l&&r<=y) return tree[id];
                int mid=l+(r-l>>1);
                int ans=0;
                if(x<=mid) ans+=realquerysum(id<<1,l,mid,x,y);
                if(y>mid) ans+=realquerysum(id<<1|1,mid+1,r,x,y);
                return ans;
            };
            return realquerysum(1,1,n,x,y);
        }
        int querykth(int k){
            function<int(int,int,int,int)> realquerykth=[&](int id,int l,int r,int k){
                if(l==r) return l;
                int mid=l+(r-l>>1);
                int lsum=mid-l+1-tree[id<<1];
                if(k<=lsum) return realquerykth(id<<1,l,mid,k);
                else return realquerykth(id<<1|1,mid+1,r,k-lsum);
            };
            return realquerykth(1,1,n,k);
        }
    };
    vector<int> fac;
    int n;
    const int mod;
    SegmentTree tree;
    Cantor(int n,int mod=9e18):n(n),mod(mod),tree(n){
        fac.resize(n+1);
        fac[0]=1;
        for(int i=1;i<=n;i++) fac[i]=fac[i-1]*i%mod;
    }
    //排名
    int get(vector<int> &v){
        int ans=0;
        int sz=v.size();
        for(int i=0;i<sz;i++){
            ans=(ans+fac[sz-i-1]*(((v[i]-1-tree.querysum(1,v[i]-1))%mod+mod)%mod)%mod)%mod;
            tree.update(v[i],1);
        }
        for(int i=0;i<sz;i++) tree.update(v[i],-1);
        return (ans+1)%mod;
    }
    //x：排名，w：位数
    vector<int> restore(int x,int w){
        vector<int> ans;
        --x;
        for(int i=w-1;i>=0;i--){
            int l=x/fac[i];
            ans.push_back(tree.querykth(l+1));
            tree.update(ans.back(),1);
            x-=l*fac[i];
        }
        for(int i=1;i<=w;i++) tree.update(i,-1);
        return ans;
    }
};
```

## 快速幂

```c++
int quickpow(int x,int y,int mod){
    if(x==0) return 0;
    int ans=1,base=x;
    while(y){
        if(y&1) ans=ans*base%mod;
        base=base*base%mod;
        y>>=1;
    }
    return ans;
}
```

## 光速幂

```c++
template<int MOD>
struct lightspeedpow{
    vector<int> powv1,powv2;
    int blocksz;
    lightspeedpow(int x,int maxn):blocksz(max(1ll,(int)sqrt(maxn))){
        powv1.resize(blocksz+2);
        powv1[0]=1;
        for(int i=1;i<=blocksz;i++){
            powv1[i]=powv1[i-1]*x%MOD;
        }
        powv2.resize(maxn/blocksz+2);
        powv2[0]=1;
        for(int i=1;i<=maxn/blocksz;i++){
            powv2[i]=powv2[i-1]*powv1[blocksz]%MOD;
        }
    }
    int operator()(int x){
        if(x<1) return 1;
        return powv2[x/blocksz]*powv1[x%blocksz]%MOD;
    }
};
```

## 模意义下大整数乘法

> 计算 $a\times b\mathrm{~mod~}p$，$a,b\leq p \leq 10^{18}$

- 龟速乘 $O(\log_2b)$

```c++
// O(log2(b))
inline ll mul(ll a, ll b, const ll p) {
    assert(p > 0);
    a %= p, b %= p;
    if (a < 0) a += p;  // 保证正数
    if (b < 0) b += p;  // 保证正数
    ll ans = 0;
    while (b) {
        if (b & 1) {
            ans += a;
            if (ans > p) ans -= p; // 直接取模会慢很多
        }
        a <<= 1;
        if (a > p) a -= p;
        b >>= 1;
    }
    return ans;
}
```

- 快速乘：$a\times b\mathrm{~mod~}p = a\times b-\lfloor a\times b/p\rfloor\times p$，可以处理模数在 `long long` 范围内、不需要使用黑科技 `__int128` 的、复杂度为 $O(1)$

```c++
ll mul(ll a, ll b, ll p) {
    ll ans = a * b - ll(1.L * a * b / p) * p;
    ans %= p;
    if (ans < 0) ans += p;
    return ans;
}
```

## 线性基

解决异或问题。原序列里的每一个数都可以由线性基里面的一些数异或得到，线性基里面任意一些数异或起来不等于0，线性基里面的数的个数唯一，且数的个数是最小的。

```cpp
struct Linear_basis {
    vector<int> p;
    vector<int> d;
    vector<int> mask; // 由哪些真实值异或而来
    vector<int> real; // 真实值
    int n;
    bool flag = 0; // 有无0
    Linear_basis(int n = 64) : n(n), p(n), mask(n), real(n) {}
    // 插入一个数
    void add(int x) {
        int _ = x;
        int msk = 0;
        for (int i = n - 1; ~i; i--) {
            if (!(x >> i))
                continue;
            if (!p[i]) {
                p[i] = x;
                real[i] = _;
                mask[i] = msk ^ (1ll << i);
                return;
            }
            x ^= p[i];
            msk ^= mask[i];
        }
        if (!x)
            flag = 1;
    }
    // 查询最大值/某个值能变成的最大值
    int querymx(int x = 0) {
        int ret = x;
        for (int i = n - 1; ~i; i--) {
            if (!(ret >> i & 1))
                ret ^= p[i];
        }
        return ret;
    }
    // 查询最小值
    int querymn() {
        if (flag)
            return 0;
        for (int i = 0; i < n; i++) {
            if (!p[i])
                continue;
            return p[i];
        }
    }
    // 询问某值，若能则返回该值如何被异或出来(msk里的每个1对应相应位置上的real)
    int query(int x) {
        int msk = 0;
        for (int i = n - 1; ~i; i--) {
            if ((x >> i) & 1) {
                msk ^= mask[i];
                x ^= p[i];
            }
        }
        if (!x)
            return msk;
        else
            return -1;
    }
    // 重构一个各个位之间互不影响的d数组
    void rebuild() {
        for (int i = n - 1; ~i; i--) {
            for (int j = i - 1; ~j; j--) {
                if (p[i] >> j & 1)
                    p[i] ^= p[j];
            }
        }
        for (int i = 0; i < n; i++) {
            if (p[i]) {
                d.push_back(p[i]);
            }
        }
    }
    // 查询第k小值(要先rebuild)
    int querykth(int k) {
        if (flag) {
            if (k == 1)
                return 0;
            k--;
        }
        int ans = 0;
        for (int i = d.size() - 1; i >= 0; i--) {
            if (k >> i & 1)
                ans ^= d[i];
        }
        return ans;
    }
};
```

## 带删线性基

```cpp
struct Linear_basis {
    vector<int> p;
    vector<int> t; // 过期时间
    int n;
    Linear_basis(int n = 64) : n(n), p(n, 0), t(n, 0) {}
    // 插入一个数x，过期时间为k时刻
    void add(int x, int k) {
        if (x == 0)
            return;
        for (int i = n - 1; ~i; --i) {
            if (!((x >> i) & 1))
                continue;
            if (p[i] == 0) {
                p[i] = x;
                t[i] = k;
                return;
            } else if (t[i] < k) {
                swap(p[i], x);
                swap(t[i], k);
            }
            x ^= p[i];
        }
    }
    // 查询time时刻x可以变成的最大值
    int querymx(int time, int x = 0) {
        for (int i = n - 1; ~i; --i) {
            if (t[i] > time && !((x >> i) & 1))
                x ^= p[i];
        }
        return x;
    }
};
```

## 扩展欧拉定理

$$
a^{b}\equiv\left\{
\begin{array}{ll}
a^{b\text{ mod}\varphi(m)}, & \text{if } \gcd(a, m)=1, \\
a^{b}, & \text{if } \gcd(a, m) \neq 1 \text{ and } b < \varphi(m), \\
a^{(b\text{ mod}\varphi(m))+\varphi(m)}, & \text{if } \gcd(a, m) \neq 1 \text{ and } b \geq \varphi(m). \\
\end{array}
\right.
$$

## 扩展欧几里德

```cpp
// ax+by=gcd(a,b)
// mod非素数，扩展欧几里得
// 返回 d=gcd(a,b);和对应于等式ax+by=d中的x,y
int exgcd(int a,int b,int &x,int &y){
    if(b==0){
        x=1,y=0;
        return a;
    }
    int x1,y1;
    int p=exgcd(b,a%b,x1,y1);
    x=y1;
    y=(x1-a/b*y1);
    return p;
}
```

## 中国剩余定理

求解如下形式的一元线性同余方程组（其中 $n_1,n_2,\cdots,n_k$ 两两互质）
$$
\left\{
\begin{array}{l}
x \equiv a_{1} \pmod{n_{1}} \\
x \equiv a_{2} \pmod{n_{2}} \\
\vdots \\
x \equiv a_{k} \pmod{n_{k}}
\end{array}
\right.
$$

```c++
i64 CRT(const vector<i64> &a, const vector<i64> &m) {
    int n = a.size();
    i64 M = 1;
    for (int i = 0; i < n; i++) M *= m[i];
    i64 res = 0;
    for (int i = 0; i < n; i++) {
        i64 Mi = M / m[i];
        i64 x, y;
        // Mi * x ≡ 1 (mod m[i]) 求逆元 <=>
        // Mi * x + m[i] * y = gcd(Mi, m[i]) = 1
        exgcd(Mi, m[i], x, y);
        // use i128 to avoid overflow !!!
        // i64 tmp = (i128)a[i] * Mi % M * x % M;
        i64 tmp = a[i] * Mi * x % M;
        res = (res + tmp) % M;
    }
    return (res + M) % M;
}
```

### 扩展CRT

不保证模数 $m_i$ 两两互质的情况。通解：$x \equiv x_0 \pmod M$

```cpp
// return: x (mod M) 最小非负解, -1无解
i64 exCRT(const vector<i64> &a, const vector<i64> &m) {
    int n = a.size();
    i64 M = m[0], R = a[0];
    for (int i = 1; i < n; i++) {
        i64 x, y;
        i64 g = exgcd(M, m[i], x, y);
        if ((a[i] - R) % g != 0) return -1;  // 无解
        // use i128 to avoid overflow !!!
        i64 t = ((i128)(a[i] - R) / g * x) % (m[i] / g);

        i128 tmp = (i128)t * M + R;
        M = (i128)M / g * m[i];  // lcm(M, m[i])
        R = (i64)((tmp % M + M) % M);
    }
    return (R % M + M) % M;
}
```

## 切比雪夫距离与曼哈顿距离之间的转化

曼哈顿意义下的坐标(x,y)，可以转化成切比雪夫意义下的(x+y,x-y)

切比雪夫意义下的坐标(x,y)，可以转化成曼哈顿意义下的坐标($\frac{x+y}{2}$,$\frac{x-y}{2}$)

## 积性函数

定义在正整数域上的函数f(x)对于任意的gcd(a,b)=1均满足f(ab)=f(a)*f(b)

可以用欧拉筛线性求值

常见积性函数

μ(n)：莫比乌斯函数
φ(n)：欧拉函数
d(n)：一个数n的约数个数
σ(n)：一个数n的约数和
f(x)=xk(k∈N)：这个玩意儿也是积性函数

## 数论分块

$\lfloor\frac n i\rfloor$块的右端点在$\lfloor\frac{n}{\lfloor\frac ni\rfloor}\rfloor$，左端点在$\lfloor\frac{n}{\lfloor\frac n{i-1}\rfloor}\rfloor+1$

$\lceil\frac ni\rceil$块的右端点在$\lfloor\frac{n-1}{\lfloor\frac{n-1}i\rfloor}\rfloor$，左端点在$\lfloor\frac{n-1}{\lfloor\frac{n-1}{i-1}\rfloor}\rfloor+1$

## 拉格朗日插值

给定 $n$ 个点值 $(x_i,y_i)$ ，确定一个最高次数为 $n-1$ 的多项式 $f(x)$ ，查询其某个单点值

通过构造 $n$ 个 $g_i(x_j)$ 使得仅当 $i=j$ 时 $g_i(x_j)=1$ ，否则 $g_i(x_j)=0$

可以构造出 $g_i(x)=\prod \limits_{i \neq j} \frac {x-x_j}{x_i-x_j}$

此时要求的 $f(x)=\Sigma_{i=0}^n y_i\prod \limits_{i \neq j} \frac {x-x_j}{x_i-x_j}$

可在 $O(n^2)$ 时间复杂度内求解

当值连续时，可以发现 $g_i(x_j)$ 的分母可以写成阶乘的形式，于是可以通过预处理将时间复杂度降至 $O(n)$

```c++
// 值不连续 O(n^2)
template <int MOD>
struct Lagrange{
    vector<pair<int, int>> v;
    int n;
    int qpow(int x, int y){
        x %= MOD;
        if (x == 0)
            return 0;
        int ans = 1, base = x;
        while (y){
            if (y & 1)
                ans = ans * base % MOD;
            base = base * base % MOD;
            y >>= 1;
        }
        return ans;
    }
    Lagrange() : n(0) {}
    void insert(int x, int y){
        n++;
        v.emplace_back(x, y);
    }
    int query(int x){
        int ret = 0;
        int t = 1;
        for (int i = 0; i < n; i++){
            if (x == v[i].first){
                return v[i].second;
            }
            t = (t * (x - v[i].first) % MOD) % MOD;
        }
        for (int i = 0; i < n; i++){
            int inv = 1;
            for (int j = 0; j < n; j++){
                if (i != j)
                    inv = inv * ((v[i].first - v[j].first) % MOD) % MOD;
            }
            ret = (ret + v[i].second * t % MOD * qpow(inv * (x - v[i].first) % MOD, MOD - 2) % MOD) % MOD;
        }
        return (ret + MOD) % MOD;
    }
};
```

```c++
//值连续 O(n)
template <int MOD>
struct Lagrange{
    vector<pair<int, int>> v;
    int n;
    int qpow(int x, int y){
        x %= MOD;
        if (x == 0)
            return 0;
        int ans = 1, base = x;
        while (y)
        {
            if (y & 1)
                ans = ans * base % MOD;
            base = base * base % MOD;
            y >>= 1;
        }
        return ans;
    }
    Lagrange() : n(0) {}
    void insert(int x, int y){
        n++;
        v.emplace_back(x, y);
    }
    int query(int x){
        int ret = 0;
        int t = 1;
        vector<int> jc(n);
        jc[0] = 1;
        for (int i = 1; i < n; i++){
            jc[i] = jc[i - 1] * i % MOD;
        }
        for (int i = 0; i < n; i++){
            if(x==v[i].first){
                return v[i].second;
            }
            t = (t * (x - v[i].first) % MOD) % MOD;
        }
        for (int i = 0; i < n; i++){
            if ((n - 1 - i) & 1)
                ret = (ret - v[i].second * t % MOD * qpow(jc[i] * jc[n - 1 - i] % MOD * ((x - v[i].first) % MOD) % MOD, MOD - 2) % MOD) % MOD;
            else
                ret = (ret + v[i].second * t % MOD * qpow(jc[i] * jc[n - 1 - i] % MOD * ((x - v[i].first) % MOD) % MOD, MOD - 2) % MOD) % MOD;
        }
        return (ret + MOD) % MOD;
    }
};
```

## 多项式

### 结论

> 已无言，不会证的都是入。
> $$
> F(x)\%(x-t)=F(t)
> $$

### 快速傅里叶变换 FFT

用法：直接当vector用，但要注意默认输入输出都是double，若要得到整数需要调用llround()

```c++
template <class T, template <class G> class Complex>
class Polynomial : public vector<T>{
    using Comp = Complex<T>;
    inline static vector<Comp> w[2];
    inline static vector<int> r;
    static void init(int _log){
        if (r.size() == (1U << _log)){
            return;
        }
        int n = 1 << _log;
        r.assign(n, 0);
        for (int i = 1; i < n; i++){
            r[i] = (r[i >> 1] >> 1) | ((i & 1) << (_log - 1));
        }
        w[0].assign(n, Comp());
        w[1].assign(n, Comp());
        const T PI = acosl(-1);
        for (int i = 0; i < n; i++){
            auto th = PI * i / n;
            auto cth = cos(1.L * th);
            auto sth = sin(1.L * th);
            w[0][i] = Comp(cth, sth);
            w[1][i] = Comp(cth, -sth);
        }
    }
    static void fft(vector<Comp> &a, int op){
        int n = a.size();
        init(__lg(n));
        for (int i = 0; i < n; i++){
            if (i < r[i]){
                swap(a[i], a[r[i]]);
            }
        }
        for (int mid = 1; mid < n; mid <<= 1){
            const int d = n / mid;
            for (int R = mid << 1, j = 0; j < n; j += R){
                for (int k = 0; k < mid; k++){
                    Comp x = a[j + k];
                    Comp y = w[op][d * k] * a[j + mid + k];
                    a[j + k] = x + y;
                    a[j + mid + k] = x - y;
                }
            }
        }
    }

public:
    using vector<T>::vector;
    constexpr friend Polynomial operator*(const Polynomial &a, const Polynomial &b){
        if (a.size() == 0 or b.size() == 0){
            return Polynomial();
        }
        int n = a.size() + b.size() - 1;
        int _log = __lg(2 * n - 1);
        int s = 1 << _log;
        if (min(a.size(), b.size()) < 128){
            Polynomial res(n);
            for (int i = 0; i < (int)a.size(); i++){
                for (int j = 0; j < (int)b.size(); j++){
                    res[i + j] += a[i] * b[j];
                }
            }
            return res;
        }
        vector<Comp> p(s), q(s);
        for (int i = 0; i < (int)a.size(); i++){
            p[i] = Comp(a[i], 0);
        }
        for (int i = 0; i < (int)b.size(); i++){
            q[i] = Comp(b[i], 0);
        }
        fft(p, 0);
        fft(q, 0);
        for (int i = 0; i < s; i++){
            p[i] = p[i] * q[i];
        }
        fft(p, 1);
        Polynomial res(n);
        for (int i = 0; i < n; i++){
            res[i] = p[i].real() / s;
        }
        return res;
    }
};
using Poly = Polynomial<double, complex>;
```

### 快速数论变换 NTT

- NTT是**整数域上的快速傅里叶变换**，用于高效求解模意义下的多项式卷积。

- 要求模数 $P$ 是形如 $k \cdot 2^n + 1$ 的质数（如 $P = 998244353 = 119 \cdot 2^{23} + 1$），且有原根 $g$；如果模数为$10^9+7$，需要**三模数 NTT + CRT**， 在 3 个 NTT 友好质数下分别做 NTT 卷积，再用 Garner/CRT 合并回 $10^9+7$。常用 3 个 NTT 质数：
  
  - $P_1=167772161=5\cdot 2^{25}+1$
  - $P_2=469762049=7\cdot 2^{26}+1$
  - $P_3=1224736769=73\cdot 2^{24}+1$

- 单位根：$w = g^{(P-1)/n} \pmod{P}$。
  
  ```c++
  Poly c = a + b;        // 加法
  Poly c = a - b;        // 减法
  Poly c = -a;           // 取负
  Poly c = a * b;        // 多项式乘法 / 卷积
  Poly c = a * k;        // 多项式乘常数
  Poly c = k * a;        // 常数乘多项式
  Poly c = a / k;        // 多项式除常数，等价于乘 k 的逆元
  ```

Poly b = a.mod(n);     // 截断，只保留前 n 项，即 mod x^n

Poly g = f.inv(n);     // 多项式求逆，满足 f * g = 1 mod x^n，要求 f[0] != 0
Poly g = f.ln(n);      // 多项式求 ln，要求 f[0] == 1
Poly g = f.exp(n);     // 多项式求 exp，要求 f[0] == 0
Poly g = f.pow(k, n);  // 多项式 k 次幂，求 f^k mod x^n，要求 k >= 0
Poly g = f.sqrt(n);    // 多项式开根，求 g^2 = f mod x^n，通常要求 f[0] == 1

Poly df = f.deriv();   // 求导
Poly F = f.integr();   // 积分，常数项为 0

auto [q, r] = f.divmod(g); // 多项式除法，f = q * g + r，要求 g 非零且 deg(r) < deg(g)

Poly g = f.mulxk(k);   // 乘 x^k，前面补 k 个 0
Poly g = f.divxk(k);   // 除 x^k，删掉前 k 项

int y = f.at(x);       // 单点评估，求 f(x)

f += g;                // f = f + g
f -= g;                // f = f - g
f *= g;                // f = f * g
f *= k;                // f = f * k
f /= k;                // f = f / k

```
```c++
const int MOD = 998244353;
const int G = 3;
int norm(int x) {
    x %= MOD;
    if (x < 0) x += MOD;
    return x;
}
int qpow(int a, int b) {
    int res = 1;
    a = norm(a);
    while (b) {
        if (b & 1) res = res * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return res;
}
int inv(int x) {
    return qpow(x, MOD - 2);
}
struct Polynomial : vector<int> {
    using Poly = Polynomial;
    using vector<int>::vector;
    inline static vector<int> w{};
    static void trim(Poly &a) {
        while (!a.empty() && a.back() == 0) a.pop_back();
    }
    static void initW(int n) {
        if ((int)w.size() >= n) return;
        w.assign(n, 0);
        w[n >> 1] = 1;
        int s = qpow(G, (MOD - 1) / n);
        for (int i = n / 2 + 1; i < n; i++) {
            w[i] = w[i - 1] * s % MOD;
        }
        for (int i = n / 2 - 1; i > 0; i--) {
            w[i] = w[i << 1];
        }
    }
    friend void dft(Poly &a) {
        int n = a.size();
        assert((n & (n - 1)) == 0);
        initW(n);
        for (int k = n >> 1; k; k >>= 1) {
            for (int i = 0; i < n; i += k << 1) {
                for (int j = 0; j < k; j++) {
                    int x = a[i + j];
                    int y = a[i + j + k];
                    a[i + j] = x + y;
                    if (a[i + j] >= MOD) a[i + j] -= MOD;
                    a[i + j + k] = (x - y + MOD) % MOD * w[k + j] % MOD;
                }
            }
        }
    }
    friend void idft(Poly &a) {
        int n = a.size();
        assert((n & (n - 1)) == 0);
        initW(n);
        for (int k = 1; k < n; k <<= 1) {
            for (int i = 0; i < n; i += k << 1) {
                for (int j = 0; j < k; j++) {
                    int x = a[i + j];
                    int y = a[i + j + k] * w[k + j] % MOD;
                    a[i + j] = x + y;
                    if (a[i + j] >= MOD) a[i + j] -= MOD;
                    a[i + j + k] = x - y;
                    if (a[i + j + k] < 0) a[i + j + k] += MOD;
                }
            }
        }
        int iv = MOD - (MOD - 1) / n;
        for (auto &x : a) x = x * iv % MOD;
        reverse(a.begin() + 1, a.end());
    }
    Poly mod(int k) const {
        Poly a = *this;
        a.resize(k);
        return a;
    }
    friend Poly operator+(const Poly &a, const Poly &b) {
        Poly c(max(a.size(), b.size()));
        for (int i = 0; i < (int)a.size(); i++) {
            c[i] += a[i];
            if (c[i] >= MOD) c[i] -= MOD;
        }
        for (int i = 0; i < (int)b.size(); i++) {
            c[i] += b[i];
            if (c[i] >= MOD) c[i] -= MOD;
        }
        return c;
    }
    friend Poly operator-(const Poly &a, const Poly &b) {
        Poly c(max(a.size(), b.size()));
        for (int i = 0; i < (int)a.size(); i++) {
            c[i] += a[i];
            if (c[i] >= MOD) c[i] -= MOD;
        }
        for (int i = 0; i < (int)b.size(); i++) {
            c[i] -= b[i];
            if (c[i] < 0) c[i] += MOD;
        }
        return c;
    }
    friend Poly operator-(Poly a) {
        for (auto &x : a) {
            if (x) x = MOD - x;
        }
        return a;
    }
    friend Poly operator*(Poly a, int b) {
        b = norm(b);
        for (auto &x : a) x = x * b % MOD;
        return a;
    }
    friend Poly operator*(int b, Poly a) {
        return a * b;
    }
    friend Poly operator/(Poly a, int b) {
        return a * ::inv(b);
    }
    friend Poly operator*(const Poly &a, const Poly &b) {
        if (a.empty() || b.empty()) return {};
        int need = a.size() + b.size() - 1;
        int n = 1 << std::__lg(2 * need - 1);
        if (((MOD - 1) & (n - 1)) || min(a.size(), b.size()) < 128) {
            Poly c(need);
            for (int i = 0; i < (int)a.size(); i++) {
                for (int j = 0; j < (int)b.size(); j++) {
                    c[i + j] = (c[i + j] + a[i] * b[j]) % MOD;
                }
            }
            return c;
        }
        Poly f = a.mod(n);
        Poly g = b.mod(n);
        dft(f), dft(g);
        for (int i = 0; i < n; i++) {
            f[i] = f[i] * g[i] % MOD;
        }
        idft(f);
        return f.mod(need);
    }
    pair<Poly, Poly> divmod(const Poly &g) const {
        Poly f = *this;
        Poly h = g;
        trim(f);
        trim(h);
        assert(!h.empty());
        int n = f.size();
        int m = h.size();
        if (n < m) {
            return {{}, f};
        }
        int qn = n - m + 1;
        Poly fr = f;
        Poly gr = h;
        reverse(fr.begin(), fr.end());
        reverse(gr.begin(), gr.end());
        fr.resize(qn);
        gr.resize(qn);
        Poly q = (fr * gr.inv(qn)).mod(qn);
        reverse(q.begin(), q.end());
        Poly r = f - q * h;
        if (m > 1) r.resize(m - 1);
        else r.clear();
        trim(q);
        trim(r);
        return {q, r};
    }
    Poly mulxk(int k) const {
        assert(k >= 0);
        Poly a = *this;
        a.insert(a.begin(), k, 0);
        return a;
    }
    Poly divxk(int k) const {
        assert(k >= 0);
        if ((int)this->size() <= k) return {};
        return Poly(this->begin() + k, this->end());
    }
    int at(int x) const {
        int ans = 0;
        x = norm(x);
        for (int i = (int)this->size() - 1; i >= 0; i--) {
            ans = (ans * x + (*this)[i]) % MOD;
        }
        return ans;
    }
    Poly &operator+=(Poly b) {
        return *this = *this + b;
    }
    Poly &operator-=(Poly b) {
        return *this = *this - b;
    }
    Poly &operator*=(Poly b) {
        return *this = *this * b;
    }
    Poly &operator*=(int b) {
        return *this = *this * b;
    }
    Poly &operator/=(int b) {
        return *this = *this / b;
    }
    Poly deriv() const {
        int n = this->size();
        if (n <= 1) return {};
        Poly f(n - 1);
        for (int i = 1; i < n; i++) {
            f[i - 1] = i * (*this)[i] % MOD;
        }
        return f;
    }
    Poly integr() const {
        int n = this->size();
        Poly f(n + 1);
        vector<int> iv(n + 1);
        if (n >= 1) iv[1] = 1;
        for (int i = 2; i <= n; i++) {
            iv[i] = iv[MOD % i] * (MOD - MOD / i) % MOD;
        }
        for (int i = 0; i < n; i++) {
            f[i + 1] = (*this)[i] * iv[i + 1] % MOD;
        }
        return f;
    }
    Poly inv(int m = -1) const {
        int n = this->size();
        m = m < 0 ? n : m;
        assert(!this->empty());
        assert((*this)[0] != 0);
        Poly p{::inv((*this)[0])};
        p.reserve(4 * m);
        for (int k = 2; k / 2 < m; k <<= 1) {
            Poly q(this->begin(), this->begin() + min(k, n));
            q = q.mod(2 * k);
            p.resize(2 * k);
            dft(q), dft(p);
            for (int i = 0; i < 2 * k; i++) {
                p[i] = p[i] * ((2 - p[i] * q[i] % MOD)%MOD + MOD) % MOD;
            }
            idft(p);
            p.resize(k);
        }
        return p.mod(m);
    }
    Poly ln(int m = -1) const {
        m = m < 0 ? this->size() : m;
        return (deriv() * inv(m)).integr().mod(m);
    }
    Poly exp(int m = -1) const {
        m = m < 0 ? this->size() : m;
        Poly p{1};
        for (int k = 1; k < m;) {
            k <<= 1;
            p = (p * (Poly{1} - p.ln(k) + mod(k))).mod(k);
        }
        return p.mod(m);
    }
    Poly pow(int k, int m = -1) const {
        m = m < 0 ? this->size() : m;
        assert(k >= 0);
        k %= MOD;
        if (k < 6) {
            Poly a = *this;
            Poly ans{1};
            while (k) {
                if (k & 1) ans = (ans * a).mod(m);
                a = (a * a).mod(m);
                k >>= 1;
            }
            return ans.mod(m);
        }
        int i = 0;
        while (i < (int)this->size() && (*this)[i] == 0) {
            i++;
        }
        if (i == (int)this->size() || __int128(i) * k >= m) {
            return Poly(m, 0);
        }
        int v = (*this)[i];
        Poly f = divxk(i) / v;
        return (f.ln(m - i * k) * k)
            .exp(m - i * k)
            .mulxk(i * k) * qpow(v, k);
    }
    Poly sqrt(int m = -1) const {
        m = m < 0 ? this->size() : m;
        Poly p{1};
        int inv2 = (MOD + 1) / 2;
        for (int k = 1; k < m;) {
            k <<= 1;
            p = (p + (mod(k) * p.inv(k)).mod(k)) * inv2;
        }
        return p.mod(m);
    }
    friend istream &operator>>(istream &is, Poly &a) {
        for (auto &x : a) {
            is >> x;
            x = norm(x);
        }
        return is;
    }
    friend ostream &operator<<(ostream &os, const Poly &a) {
        for (int i = 0; i < (int)a.size(); i++) {
            os << a[i] << " \n"[i + 1 == (int)a.size()];
        }
        return os;
    }
};
using Poly = Polynomial;
```

### 快速沃尔什变换 FWT

变换一次复杂度 $\mathcal{O}(n\log n)$，做一次点乘 $\mathcal{O}(n)$
一个完整的 FWT 过程是先做一次正变换，再做一次点乘，再做一次逆变换，复杂度 $\mathcal{O}(n\log n)$
你也可以应用如下的式子做单点的变换，时间复杂度 $\mathcal{O}(n)$

#### OR

正变换:
$$
\hat{a}[s] = \sum_{x\subseteq s} a[x]
$$
逆变换:
$$
a[s] = \sum_{x\subseteq s} (-1)^{|s|-|x|}\hat{a}[x]
$$

#### AND FWT

正变换:
$$
\hat{a}[s] = \sum_{x\supseteq s} a[x]
$$
逆变换:
$$
a[s] = \sum_{x\supseteq s} (-1)^{|x|-|s|}\hat{a}[x]
$$

#### XOR

正变换:
$$
\hat{a}[s] = \sum_{x=0}^{N-1} a[x]\cdot (-1)^{popcount(s\&x)}
$$
逆变换:
$$
a[x] = \frac{1}{N}\sum_{s=0}^{N-1} \hat{a}[s]\cdot (-1)^{popcount(s\&x)}
$$

```c++
template <int MOD>
struct FWT {
    enum class Type { OR, AND, XOR };

    static inline int add(int a, int b) {
        a += b;
        if (a >= MOD) a -= MOD;
        return a;
    }
    static inline int sub(int a, int b) {
        a -= b;
        if (a < 0) a += MOD;
        return a;
    }
    static inline int mul(int a, int b) {
        return 1LL * a * b % MOD;
    }
    // 只有 XOR 需要求逆元
    static inline int pow(int a, ll e) {
        ll r = 1;
        while (e) {
            if (e & 1) r = mul(r, a);
            a = mul(a, a);
            e >>= 1;
        }
        return (int)r;
    }

    // FWT 就地变换: opt = 1 为正变换，opt = -1 为逆变换
    static void transform(vector<int>& a, Type type, int opt) {
        const int n = a.size();
        for (int len = 1; len < n; len <<= 1) {
            for (int i = 0; i < n; i += (len << 1)) {
                for (int j = 0; j < len; j++) {
                    int &x = a[i + j], &y = a[i + j + len];
                    if (type == Type::OR) {
                        // forward: (x, y) -> (x, x+y)
                        // inverse: (x, y) -> (x, y-x)
                        if (opt == 1) {
                            y = add(y, x);
                        } else {
                            y = sub(y, x);
                        }
                    } else if (type == Type::AND) {
                        // forward: (x, y) -> (x+y, y)
                        // inverse: (x, y) -> (x-y, y)
                        if (opt == 1) {
                            x = add(x, y);
                        } else {
                            x = sub(x, y);
                        }
                    } else if (type == Type::XOR) {
                        // (x, y) -> (x+y, x-y) ；逆变换后再整体乘 1/n
                        int u = x, v = y;
                        x = add(u, v);
                        y = sub(u, v);
                    }
                }
            }
        }
        // xor 逆变换，归一化，所有元素乘 1/n = (1/2)^{log2 n}
        if (type == Type::XOR && opt == -1) {
            const int inv2 = (MOD + 1) / 2;
            int k = __builtin_ctz(n);  // log2(n)
            int invn = pow(inv2, k);   // (1/2)^k
            for (int& x : a) {
                x = mul(x, invn);
            }
        }
    }

    // 点乘
    static void pointwise(vector<int>& a, const vector<int>& b) {
        const int n = a.size();
        for (int i = 0; i < n; ++i) {
            a[i] = mul(a[i], b[i]);
        }
    }

    // 卷积（OR / AND / XOR）
    static vector<int> convolution(vector<int> a, vector<int> b, Type type) {
        // 需要 A、B 长度相等且为 2 的幂；若需要，也可在外部补零到 2^k
        assert(a.size() == b.size());
        assert((a.size() & (a.size() - 1)) == 0);  // 检查是否为 2 的幂

        transform(a, type, 1);   // 正变换
        transform(b, type, 1);   // 正变换
        pointwise(a, b);         // 点乘
        transform(a, type, -1);  // 逆变换
        return a;
    }
};

using fwt = FWT<998244353>;

int main() {
    int n; cin >> n;
    int N = 1 << n;  // 需要保证长度是2的幂次，否则补0
    vector<int> a(N), b(N);
    for (int i = 0; i < N; i++) cin >> a[i];
    for (int i = 0; i < N; i++) cin >> b[i];

    auto OR = fwt::convolution(a, b, fwt::Type::OR);
    auto AND = fwt::convolution(a, b, fwt::Type::AND);
    auto XOR = fwt::convolution(a, b, fwt::Type::XOR);

    for (int i = 0; i < N; i++) cout << OR[i] << " \n"[i == N - 1];
    //...
}
```

## 求n!质因数p的个数

$[1,n]$ 每个数都可以贡献质因数 $p$

$p$ 的倍数贡献一个，$p^2$ 的倍数继续额外贡献一个

$cnt=\left\lfloor \frac{n}{p} \right\rfloor+\left\lfloor \frac{n}{p^2} \right\rfloor+...+\left\lfloor \frac{n}{p^k} \right\rfloor$

## 二次剩余（模意义下开根）

$gcd(n,p)=1$

$x^2=n(mod~p)$

若存在 $x$，则 $n$ 为模 $p$ 的二次剩余（$p$ 是奇素数），否则就是二次非剩余

### Euler判定

a是p的二次剩余当且仅当

$n^\frac{p-1}{2} \equiv1(mod~p)\\$

是p的二次非剩余当且仅当

$n^\frac{p-1}{2} \equiv-1(mod~p)\\$

> 如果p%4=3
> 
> $n^{(p+1)/4} mod~p$是一个解

这个方程**只有两个解**且它们**互为相反数**。一个二次剩余对应一对模意义下不同的相反数（*p* 为奇素数所以相反数的奇偶性不同）。因为模 *p* 意义下可以找到 $\frac{p-1}{2}$对非零的相反数，所以在模 *p* 意义下共有 $\frac{p-1}{2}$ 个二次剩余。

### Cipolla算法

先找到一个 *a* 使得 $a^2-n$ 是二次非剩余，令 $w^2\equiv a^2-n(mod~p)$ ,则 $(a+w)^\frac{p+1}{2}$ 即是方程的一个解，其相反数则是另一个解。

在[0,p)中随机一个a，并用欧拉准则判断$a^2-n$是否是二次非剩余，因为二次非剩余有p/2个，所以期望两次找到。

```cpp
int quickpow(int x,int y,int mod){
    int ans=1,base=x;
    x%=mod;
    if(x==0) return 0;
    while(y){
        if(y&1) ans=ans*base%mod;
        base=base*base%mod;
        y>>=1;
    }
    return ans;
}
pair<int,int> Cipolla(int n,int mod){
    if(n==0) return make_pair(0,0);
    n%=mod;
    if(quickpow(n,(mod-1)/2,mod)==mod-1) return {-1,-1}; // 二次非剩余，无解
    int a=0,w;
    while(1){
        a=rand()%mod;
        w=((a*a%mod-n)%mod+mod)%mod;
        if(quickpow(w,(mod-1)/2,mod)==mod-1) break;
    }
    auto mulcomplex=[&](pair<int,int> x,pair<int,int> y,int mod){
        return make_pair(((x.first*y.first%mod+w*x.second%mod*y.second%mod)%mod+mod)%mod,(x.first*y.second%mod+x.second*y.first%mod)%mod);
    };
    auto quickpowcomplex=[&](pair<int,int> x,int y,int mod){
        pair<int,int> base=x,ans={1,0};
        while(y){
            if(y&1) ans=mulcomplex(ans,base,mod);
            base=mulcomplex(base,base,mod);
            y>>=1;
        }
        return ans;
    };
    pair<int,int> result;
    result.first=quickpowcomplex(make_pair(a,1),(mod+1)/2,mod).first;
    result.second=(-result.first%mod+mod)%mod;
    if(result.first>result.second) swap(result.first,result.second);
    return result; // 返回两个根并排好序
}
```

## 杜教筛

- 亚线性时间复杂度内计算**数论函数的前缀和**：通过狄利克雷卷积和整除分块，将一个复杂的前缀和问题转化为递归求解的形式。
- 求解 $S(n)=\sum_{i=1}^n{f(i)}$，其中 $f$ 是数论函数，且满足：能找到另一个数论函数 $g$，使得 $f*g$ 的前缀和，以及 $g$ 本身的前缀和，能够快速计算（通常是 $O(1)$ 或 $O(\sqrt{n})$）
- 最优时间复杂度：$O(N^{2/3})$，预处理前 $M$ 个点的前缀和（`maxn`），总复杂度 $O(M+N/\sqrt{M})$，当 $M=N^{2/3}$ 时达到最优。

关键在于，构造出一个能快速计算的 $g(n)$ 与 $(f*g)(n)$ ，用 $g(1)S(n)=\Sigma^n_{i=1}(f*g)(i)-\Sigma^n_{y=2}g(y)S(\lfloor \frac ny\rfloor)$ 递归计算

```cpp
#define int long long

struct Sieve {
    int n;
    vector<int> prime;
    vector<int> spf;      // smallest prime factor
    vector<int> mu;       // mobius
    vector<int> phi;      // euler
    vector<int> mu_pre;   // prefix sum of mobius
    vector<int> phi_pre;  // prefix sum of euler

    Sieve(int _n) : n(_n) {
        prime.reserve(n / (log(n) - 1.1));
        spf.resize(n + 1, 0);
        mu.resize(n + 1, 0);
        phi.resize(n + 1, 0);
        mu_pre.resize(n + 1, 0);
        phi_pre.resize(n + 1, 0);
        mu[1] = 1;
        phi[1] = 1;

        for (int i = 2; i <= n; i++) {
            if (spf[i] == 0) {
                prime.push_back(i);
                spf[i] = i;
                phi[i] = i - 1;
                mu[i] = -1;
            }

            for (int p : prime) {
                if (i * p > n) break;
                spf[i * p] = p;
                if (i % p == 0) {
                    mu[i * p] = 0;
                    phi[i * p] = phi[i] * p;
                    break;
                } else {
                    mu[i * p] = -mu[i];
                    phi[i * p] = phi[i] * (p - 1);
                }
            }
        }

        for (int i = 1; i <= n; i++) {
            mu_pre[i] = mu_pre[i - 1] + mu[i];
            phi_pre[i] = phi_pre[i - 1] + phi[i];
        }
    }

    int get_phi_pre(int n) const { return phi_pre[n]; }
    int get_mu_pre(int n) const { return mu_pre[n]; }
};

class DjSieve {
    const Sieve &base;
    unordered_map<int, int> sphi;  // prefix sum of euler
    unordered_map<int, int> smu;   // prefix sum of mobius

public:
    explicit DjSieve(const Sieve &sieve) : base(sieve) {}

    int get_phi_sum(int n) {
        if (n <= base.n) return base.get_phi_pre(n);
        auto it = sphi.find(n);
        if (it != sphi.end()) return it->second;

        int res = n * (n + 1) / 2;
        for (int l = 2, r; l <= n; l = r + 1) {
            r = n / (n / l);
            res -= (r - l + 1) * get_phi_sum(n / l);
        }
        return sphi[n] = res;
    }

    int get_mu_sum(int n) {
        if (n <= base.n) return base.get_mu_pre(n);
        auto it = smu.find(n);
        if (it != smu.end()) return it->second;

        int res = 1;
        for (int l = 2, r; l <= n; l = r + 1) {
            r = n / (n / l);
            res -= (r - l + 1) * get_mu_sum(n / l);
        }
        return smu[n] = res;
    }
};

constexpr int maxn = 1e6 + 5;  // n^(2/3)
Sieve sieve(maxn);
DjSieve dj(sieve);
```

## Min25筛

在 $O\left(\frac{n^{3/4}}{\log n}\right)$ 时间内求出积性函数f的前缀和，且需要满足

- $f(x)$ 在 $x$ 为质数的时候有一个简单多项式表示

- $f(x^c)$ 在 $x$ 为质数的时候可以快速计算

f：原函数（积性函数）

fpi：新函数（完全积性函数），质数处与f取值相同

1. 先求出 $g[n][i]$，即 $x\in[2,n]$ 且 ( $x$ 为质数或 $x$ 的最小质因数 $>prime_i$ ) 时，所有fpi(x)的和
2. $s[n][i]$ 表示，$x\in[2,n]$ 且最小质因数 $>prime_i$ 时，所有 $f(x)$ 的和

$g(n, j) = g(n, j-1) - fpi(p_j) \left( g\left(\frac{n}{p_j}, j-1\right) - g(p_{j-1}, j-1) \right)$

$S(n, x) = g(n) - sp_x + \sum_{\substack{p_k^e \leq n \\ k > x}} f(p_k^e) \left( S\left(\frac{n}{p_k^e}, k\right) + [e \neq 1] \right)$

即质数贡献（可能需要对某些值进行修改）+合数贡献

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int MOD=1e9+7;
int f(int ind,int x){
    x%=MOD;
    if(ind==0){
        return x;
    }else{
        return x*x%MOD;
    }
}
int fpi(int ind,int x){
    x%=MOD;
    if(ind==0){
        return x;
    }else{
        return x*x%MOD;
    }
}
int inv6=166666668,inv2=500000004;
int fpisum(int ind,int x){
    x%=MOD;
    if(ind==0){
        return (((1+x)%MOD*x%MOD*inv2%MOD-1)%MOD+MOD)%MOD;
    }else{
        return ((x%MOD*(x+1)%MOD*(2*x%MOD+1)%MOD*inv6%MOD-1)%MOD+MOD)%MOD;
    }
}
template<int items>
struct Min25{
    struct EulerSieve{
        vector<int> prime;
        vector<int> v;
        vector<array<int,items>> sp;
        int n;
        EulerSieve(int n,const int MOD):v(n+1){
            this->n=n;
            for(int i=2;i<=n;i++){
                if(v[i]==0){
                    prime.push_back(i);
                    v[i]=i;
                    sp.emplace_back();
                    for(int j=0;j<items;j++){
                        if(sp.size()>=2) sp.back()[j]=sp[sp.size()-2][j];
                        sp.back()[j]=(sp.back()[j]+fpi(j,i))%MOD;
                    }
                }
                for(int &p:prime){
                    if(i*p>n) break;
                    v[i*p]=p;
                    if(i%p==0) break;
                }
            }
        }
    };
    int n,sqr;
    EulerSieve eulersieve;
    vector<int> ind1,ind2,w;
    vector<array<int,items>> g;
    int result=0;
    //f(i,x) 多项式拆成若干个单项式，第i个单项式(x=p^t)
    //fpi(i,x) 多项式拆成若干个单项式，和f[i]相等的完全积性函数
    //fpisum(i,x) 多项式拆成若干个单项式，fpi[i](x)前缀和(不含fpi[i][1])
    //sign[i]第i个单项式的系数（符号） 
    Min25(int n,array<int,items> &sign):n(n),sqr(sqrt(n)),ind1(sqr+10),ind2(sqr+10),eulersieve(sqr,MOD){
        w.reserve(2*sqr+10);
        g.reserve(2*sqr+10);
        ind1.reserve(sqr+10);
        ind2.reserve(sqr+10);
        for(int i=1;i<=n;){
            int j=n/(n/i);
            w.emplace_back(n/i);
            g.emplace_back();
            for(int j=0;j<items;j++){
                g.back()[j]=fpisum(j,w.back());
            }
            if(n/i<=sqr) ind1[n/i]=w.size()-1;
            else ind2[n/(n/i)]=w.size()-1;
            i=j+1;
        }
        for(int t=0;t<eulersieve.prime.size();t++){
            int p=eulersieve.prime[t];
            for(int i=0;i<w.size()&&p<=w[i]/p;i++){
                int k=w[i]/p<=sqr?ind1[w[i]/p]:ind2[n/(w[i]/p)];
                for(int j=0;j<items;j++){
                    g[i][j]-=(fpi(j,p)*(g[k][j]%MOD-(t-1>=0?eulersieve.sp[t-1][j]:0ll))%MOD)%MOD;
                    g[i][j]=(g[i][j]%MOD+MOD)%MOD;
                }
            }
        }
        auto s=[&](auto &&self,int x,int y){
            if(y>=1&&eulersieve.prime[y-1]>=x) return 0ll;
            int k=x<=sqr?ind1[x]:ind2[n/x];
            int ans=0;
            for(int i=0;i<items;i++){
                int t=y>0?eulersieve.sp[y-1][i]:0;
                ans=(ans+sign[i]*(g[k][i]-t)%MOD)%MOD;
            }
            for(int i=y+1;i<=eulersieve.prime.size()&&eulersieve.prime[i-1]<=x/eulersieve.prime[i-1];i++){
                int pe=eulersieve.prime[i-1];
                for(int e=1;pe<=x;e++,pe*=eulersieve.prime[i-1]){
                    int xx=pe%MOD;
                    int sum=0;
                    for(int j=0;j<items;j++){
                        sum=(sum+sign[j]*f(j,pe))%MOD;
                    }
                    ans=(ans+sum*(self(self,x/pe,i)+(e!=1))%MOD)%MOD;
                    ans=(ans+MOD)%MOD;
                }
            }
            return ans;
        };
        result=((s(s,n,0)+1)%MOD+MOD)%MOD;
    }
    int get(){
        return result;
    }
};
void solve(){
    int n;
    cin>>n;
    array<int,2> sign={-1,1};
    Min25<2> min25(n,sign);
    cout<<min25.get()<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int MOD=1e9+7;
int f(int x,int t){
    return (x^t)%MOD;
}
int fpi(int ind,int x){
    x%=MOD;
    if(ind==0){
        return x;
    }else{
        return 1;
    }
}
int inv2=500000004;
int fpisum(int ind,int x){
    x%=MOD;
    if(ind==0){
        return (((1+x)%MOD*x%MOD*inv2%MOD-1)%MOD+MOD)%MOD;
    }else{
        return ((x-1)%MOD+MOD)%MOD;
    }
}
template<int items>
struct Min25{
    struct EulerSieve{
        vector<int> prime;
        vector<int> v;
        vector<array<int,items>> sp;
        int n;
        EulerSieve(int n,const int MOD):v(n+1){
            this->n=n;
            for(int i=2;i<=n;i++){
                if(v[i]==0){
                    prime.push_back(i);
                    v[i]=i;
                    sp.emplace_back();
                    for(int j=0;j<items;j++){
                        if(sp.size()>=2) sp.back()[j]=sp[sp.size()-2][j];
                        sp.back()[j]=(sp.back()[j]+fpi(j,i))%MOD;
                    }
                }
                for(int &p:prime){
                    if(i*p>n) break;
                    v[i*p]=p;
                    if(i%p==0) break;
                }
            }
        }
    };
    int n,sqr;
    EulerSieve eulersieve;
    vector<int> ind1,ind2,w;
    vector<array<int,items>> g;
    int result=0;
    //f[i](x) 多项式拆成若干个单项式，第i个单项式(x=p^t)
    //fpi[i](x) 多项式拆成若干个单项式，和f[i]相等的完全积性函数
    //fpisum[i](x) 多项式拆成若干个单项式，fpi[i](x)前缀和(不含fpi[i][1])
    //sign[i]第i个单项式的系数（符号） 
    Min25(int n,array<int,items> &sign):n(n),sqr(sqrt(n)),ind1(sqr+10),ind2(sqr+10),eulersieve(sqr,MOD){
        w.reserve(2*sqr+10);
        g.reserve(2*sqr+10);
        ind1.reserve(sqr+10);
        ind2.reserve(sqr+10);
        for(int i=1;i<=n;){
            int j=n/(n/i);
            w.emplace_back(n/i);
            g.emplace_back();
            for(int j=0;j<items;j++){
                g.back()[j]=fpisum(j,w.back());
            }
            if(n/i<=sqr) ind1[n/i]=w.size()-1;
            else ind2[n/(n/i)]=w.size()-1;
            i=j+1;
        }

        for(int t=0;t<eulersieve.prime.size();t++){
            int p=eulersieve.prime[t];
            for(int i=0;i<w.size()&&p<=w[i]/p;i++){
                int k=w[i]/p<=sqr?ind1[w[i]/p]:ind2[n/(w[i]/p)];
                for(int j=0;j<items;j++){
                    g[i][j]-=(fpi(j,p)*(g[k][j]%MOD-(t-1>=0?eulersieve.sp[t-1][j]:0ll))%MOD)%MOD;
                    g[i][j]=(g[i][j]%MOD+MOD)%MOD;
                }
            }
        }
        auto s=[&](auto &&self,int x,int y){
            if(y>=1&&eulersieve.prime[y-1]>=x) return 0ll;
            int k=x<=sqr?ind1[x]:ind2[n/x];
            int ans=0;
            for(int i=0;i<items;i++){
                int t=y>0?eulersieve.sp[y-1][i]:0;
                ans=(ans+sign[i]*(g[k][i]-t)%MOD)%MOD;
            }
            if(x>=2&&y==0){
                ans=(ans+2)%MOD;
            }
            for(int i=y+1;i<=eulersieve.prime.size()&&eulersieve.prime[i-1]<=x/eulersieve.prime[i-1];i++){
                int pe=eulersieve.prime[i-1];
                for(int e=1;pe<=x;e++,pe*=eulersieve.prime[i-1]){
                    int xx=pe%MOD;
                    int sum=f(eulersieve.prime[i-1],e);
                    ans=(ans+sum*(self(self,x/pe,i)+(e!=1))%MOD)%MOD;
                    ans=(ans+MOD)%MOD;
                }
            }
            return ans;
        };
        result=((s(s,n,0)+1)%MOD+MOD)%MOD;
    }
    int get(){
        return result;
    }
};
void solve(){
    int n;
    cin>>n;
    if(n==1){
        cout<<1<<"\n";
        return;
    }else if(n==2){
        cout<<4<<"\n";
        return;
    }
    array<int,2> sign={1,-1};
    Min25<2> min25(n,sign);
    cout<<min25.get()<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 离线算法

### 莫队

#### 树上莫队

**分块大小为sqrt(2n)最优**

```
题目大意：给出一棵树，树的每个节点有一个字母，有m个询问，每个询问给出两节点的编号u，v，回答两个节点之间的简单路径所包含的字母是否可以通过重新排列组合成回文串（所有字母都要用上）。

Format：

Input:第一行输出n，m，n代表树的节点数，m代表询问的次数，第二行输入一个字符串（均为小写字母），第i个字母表示第i个节点的字母是什么，接下来n-1行每行给出u，v表示一条边。接下来m行每行给出一堆节点编号u，v代表一次询问

Out：如果可以组成回文串，输出yes，否则输出no

Samples：

5 5
abaac
1 2
2 3
2 4
4 5
1 3
2 4
1 5
1 4
5 5

yes
no
no
yes
yes
```

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct edge{
    int x,y,id;
    bool needLCA;
};
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int n,m;
    string s;
    cin>>n>>m;
    cin>>s;
    s=" "+s;
    vector<vector<int>> v(n+1);
    for(int i=1;i<=n-1;i++){
        int x,y;
        cin>>x>>y;
        v[x].push_back(y);
        v[y].push_back(x);
    }
    vector<int> dep(n+1,0),cnt,in(n+1),out(n+1),dfscnt(2*n+5);
    vector<vector<int>> fa(n+1,vector<int>(21));
    function<void(int,int)> dfs=[&](int x,int father){
        dep[x]=dep[father]+1;
        fa[x][0]=father;
        for(int i=1;i<=20;i++){
            fa[x][i]=fa[fa[x][i-1]][i-1];
        }
        in[x]=cnt.size();
        dfscnt[in[x]]=x;
        cnt.push_back(x);
        for(int &p:v[x]){
            if(p==father) continue;
            dfs(p,x);
        }
        out[x]=cnt.size();
        dfscnt[out[x]]=x;
        cnt.push_back(x);
    };
    function<int(int,int)> LCA=[&](int x,int y){
        if(dep[x]<dep[y]) swap(x,y);
        for(int i=20;i>=0;i--){
            if(dep[fa[x][i]]>=dep[y]) x=fa[x][i];
        }
        if(x==y) return x;
        for(int i=20;i>=0;i--){
            if(fa[x][i]!=fa[y][i]){
                x=fa[x][i];
                y=fa[y][i];
            }
        }
        return fa[x][0];
    };
    dfs(1,0);
    vector<edge> query(m);
    for(int i=0;i<m;i++){
        cin>>query[i].x>>query[i].y;
        query[i].id=i;
        int t=LCA(query[i].x,query[i].y);
        if(t==query[i].x||t==query[i].y){
            query[i].needLCA=0;
            query[i].x=in[query[i].x];
            query[i].y=in[query[i].y];
            if(query[i].x>query[i].y) swap(query[i].x,query[i].y);
        }else{
            query[i].needLCA=1;
            if(out[query[i].x]>in[query[i].y]) swap(query[i].x,query[i].y);
            query[i].x=out[query[i].x];
            query[i].y=in[query[i].y];
        }
    }
    const int Blocksize=sqrt(2*n);
    vector<int> chcnt(26,0);
    sort(query.begin(),query.end(),[&](edge &a,edge &b){
        if(a.x/Blocksize!=b.x/Blocksize) return a.x/Blocksize<b.x/Blocksize;
        else if(a.x/Blocksize%2==0) return a.y<b.y;
        else return a.y>b.y;
    });
    int l=1,r=0;
    vector<bool> use(n+1,0);
    auto upd=[&](int x){
        use[dfscnt[x]]=!use[dfscnt[x]];
        if(use[dfscnt[x]]) chcnt[s[dfscnt[x]]-'a']++;
        else chcnt[s[dfscnt[x]]-'a']--;
    };
    vector<bool> ans(m);
    auto getans=[&](){
        int a=0,b=0;
        for(int i=0;i<26;i++){
            if(chcnt[i]==0) continue;
            if(chcnt[i]%2==0) a++;
            else b++;
        }
        return b<=1;
    };
    for(int i=0;i<m;i++){
          while(l>query[i].x){
            upd(--l);
        }    
          while(r<query[i].y){
            upd(++r);
        }
        while(l<query[i].x){
            upd(l++);
        }
        while(r>query[i].y){
            upd(r--);
        }
        if(query[i].needLCA){
            int t=LCA(dfscnt[query[i].x],dfscnt[query[i].y]);
            upd(in[t]);
            ans[query[i].id]=getans();
            upd(in[t]);
        }else ans[query[i].id]=getans();
    }
    for(auto p:ans){
        cout<<(p?"yes":"no")<<"\n";
    }
}
```

#### 回滚莫队&不删除莫队

只增加或者只删除，其中分块大小取n/sqrt(m)最优，n为数组长度，m为询问次数

为了防止分块大小等于0，可以把分块大小取为max(1,n/sqrt(m));

```
给定一个序列，多次询问一段区间[l,r]，求区间中相同的数的最远间隔距离
序列中两个元素的间隔距离指的是两个元素下标差的绝对值
第一行一个整数𝑛，表示序列长度。第二行𝑛个整数，描述这个序列。第三行一个整数𝑚，表示询问个数。之后 𝑚行，每行两个整数 l,r 表示询问区间。输出m行，表示答案，如果区间内不存在两个数相同，输出0
输入：
8
1 6 2 2 3 3 1 6
5
1 4
2 5
2 8
5 6
1 7
输出：
1
1
6
1
6
```

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct edge{
    int l,r,id,blockid;
};
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int n;
    cin>>n;
    vector<int> a(n+1),num;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        num.push_back(a[i]);
    }
    sort(num.begin(),num.end());
    num.erase(unique(num.begin(),num.end()),num.end());
    auto getid=[&](int x){
        return lower_bound(num.begin(),num.end(),x)-num.begin();
    };
    for(int i=1;i<=n;i++) a[i]=getid(a[i]);
    int m;
    cin>>m;
    const int blocksize=max(1ll,(int)(n/sqrt(m)));
    vector<edge> query(m);
    for(int i=0;i<m;i++){
        query[i].id=i;
        cin>>query[i].l>>query[i].r;
        query[i].blockid=query[i].l/blocksize;
    }
    sort(query.begin(),query.end(),[](edge a,edge b){
        if(a.blockid!=b.blockid) return a.blockid<b.blockid;
        else return a.r<b.r;
    });
    int lastblock=-1,r=-1;
    vector<int> cntright(num.size(),-1),cntleft(num.size(),-1),cnt(num.size(),-1);
    vector<bool> cntback(num.size(),0);
    vector<int> ans(m);
    int maxn=0;
    for(int i=0;i<m;i++){
        if(query[i].l/blocksize==query[i].r/blocksize){
            int tmp=0;
            for(int j=query[i].l;j<=query[i].r;j++){
                if(cnt[a[j]]==-1) cnt[a[j]]=j;
                else tmp=max(tmp,j-cnt[a[j]]);
            }
            ans[query[i].id]=tmp;
            for(int j=query[i].l;j<=query[i].r;j++){
                cnt[a[j]]=-1;
            }
        }else{
            if(lastblock!=query[i].blockid){
                if(lastblock!=-1){
                    for(int j=(lastblock+1)*blocksize;j<=r;j++){
                        cntleft[a[j]]=cntright[a[j]]=-1;
                    }
                }
                lastblock=query[i].blockid;
                r=(lastblock+1)*blocksize-1;
                maxn=0;
            }
            while(r<query[i].r){
                ++r;
                if(cntright[a[r]]==-1){
                    cntleft[a[r]]=cntright[a[r]]=r;
                }else{
                    cntright[a[r]]=r;
                    maxn=max(maxn,r-cntleft[a[r]]);
                }
            }
            int maxnback=maxn;
            for(int j=(lastblock+1)*blocksize-1;j>=query[i].l;j--){
                if(cntright[a[j]]==-1){
                    cntback[a[j]]=1;
                    cntright[a[j]]=j;
                }else{
                    maxn=max(maxn,cntright[a[j]]-j);
                }
            }
            ans[query[i].id]=maxn;
            maxn=maxnback;
            for(int j=(lastblock+1)*blocksize-1;j>=query[i].l;j--){
                if(cntback[a[j]]){
                    cntright[a[j]]=-1;
                    cntback[a[j]]=0;
                }
            }
        }
    }
    for(int &p:ans) cout<<p<<"\n";
}
```

## 异或哈希

### Zobrist Hash

用于棋盘状态压缩，每个位置的每个棋子状态（例如（1,1）位置为黑棋）使用 `mt19937_64` 赋予一个随机值，最后整个棋盘的状态等于所有棋子的异或和

## 狄利克雷卷积

定义
$$
(f * g)(n) = \sum_{k \mid n} f(k) g\left(\frac{n}{k}\right) = \sum_{k \ell = n} f(k) g(\ell).
$$
常见卷积

1. 单位函数 $\varepsilon$ 是莫比乌斯函数 $\mu$ 和常数函数 $1$ 的 Dirichlet 卷积：
   
   $$
   \varepsilon = \mu * 1 \quad \Longleftrightarrow \quad \varepsilon(n) = \sum_{d \mid n} \mu(d).
   $$

2. 除数个数函数 $\tau$ 是常数函数 $1$ 和它自身的 Dirichlet 卷积：
   
   $$
   \tau = 1 * 1 \quad \Longleftrightarrow \quad \tau(n) = \sum_{d \mid n} 1.
   $$

3. 除数和函数 $\sigma$ 是恒等函数 $\operatorname{id}$ 和常数函数 $1$ 的 Dirichlet 卷积：
   
   $$
   \sigma = \operatorname{id} * 1 \quad \Longleftrightarrow \quad \sigma(n) = \sum_{d \mid n} d.
   $$

4. 欧拉函数 $\varphi$ 是恒等函数 $\operatorname{id}$ 和莫比乌斯函数 $\mu$ 的 Dirichlet 卷积：
   
   $$
   \varphi = \operatorname{id} * \mu \quad \Longleftrightarrow \quad \varphi(n) = \sum_{d \mid n} d \cdot \mu\left( \frac{n}{d} \right).
   $$

# 图论

## Cayley 凯莱定理

一个完全图不同的生成树的数量有$n^{n-2}$种

## 最小生成树

### Kruskal

时间复杂度$O（mlogm）$

```c++
struct edge{
    int x,y,k;
};
vector<edge> v(n+1);
DSU dsu(n);
int kruskal(){
    int ans=0,t=0;
    for(int i=0;i<v.size();i++){
        int fax=dsu.find(v[i].x);
        int fay=dsu.find(v[i].y);
        if(fax==fay) continue;
        merge(fax,fay);
        ans+=v[i].k;
        ++t;
        if(t==n-1){
            return ans;
        }
    }
    return -1;
}
```

## 判负环

SPFA判负环，一旦一个点入队次数大于等于n，即存在负环，时间复杂度O(nm)

```c++
const int INF=1e18;
//bfs SPFA判是否存在经过s的负环
//1表示存在负环，0表示不存在负环
    auto SPFA=[&](int s){
        vector<bool> vis(n+1,0);
        vector<int> dis(n+1,INF);
        vector<int> in(n+1,0);
        queue<int> q;
        dis[s]=0;
        q.push(s);
        while(!q.empty()){
            int f=q.front();
            q.pop();
            vis[f]=0;
            for(auto &[to,w]:v[f]){
                if(dis[to]>dis[f]+w){
                    dis[to]=dis[f]+w;
                    if(!vis[to]){
                          if(++in[to]>n) return 1;
                        vis[to]=1;
                        q.push(to);
                    }
                }
            }
        }
        return 0;
    };
```

```c++
//存在负环返回1，否则返回0
    auto check=[&](){
        bool flag=0;
        vector<int> dis(n+1,0);
        vector<bool> vis(n+1,0);
        function<int(int)> dfs=[&](int x){
            if(vis[x]){
                flag=1;
                return 1;
            }
            vis[x]=1;
            for(auto &[to,w]:v[x]){
                if(dis[to]>dis[x]+w){
                    dis[to]=dis[x]+w;
                    dfs(to);
                    if(flag) return 1;
                }
            }
            vis[x]=0;
        };
        for(int i=1;i<=n;i++){
            dfs(i);
            if(flag) return 1;
        }
        return 0;
    };
```

## 网络流

网络指一个特殊的有向图G=(V,E)，其与一般有向图的不同之处在于有容量和源汇点（源点s，汇点t）。任意节点净流量为0，且流经该边的流量不得超过该边的容量（边(u,v)的容量记作c(u,v)）。定义f的流量为源点s的净流量。

### 最大流

使流量f尽可能大，dinic算法，时间复杂度$O(mn^2)$

二分图最大匹配时间复杂度$O(m\sqrt{n})$

```c++
struct Flow{
    const int n;
    const int MAXN=1e18;
    vector<pair<int,int>> e;
    vector<vector<int>> g;
    vector<int> cur,dep;
    Flow(int n):n(n),g(n+1){}
    bool bfs(int s,int t){
        dep.assign(n+1,-1);
        queue<int> q;
        dep[s]=0;
        q.push(s);
        while(!q.empty()){
            const int u=q.front();
            q.pop();
            for(int i:g[u]){
                auto [v,c]=e[i];
                if(c>0&&dep[v]==-1){
                    dep[v]=dep[u]+1;
                    if(v==t) return 1;
                    q.push(v);
                }
            }
        }
        return 0;
    }
    int dfs(int u,int t,int f){
        if(u==t) return f;
        int res=f;
        for(int &i=cur[u];i<g[u].size();i++){
            const int j=g[u][i];
            auto [v,c]=e[j];
            if(c>0&&dep[v]==dep[u]+1){
                int out=dfs(v,t,min(res,c));
                e[j].second-=out;
                e[j^1].second+=out;
                res-=out;
                if(res==0) return f;
            }
        }
        return f-res;
    }
    void add(int u,int v,int c){
        g[u].push_back(e.size());
        e.emplace_back(v,c);
        g[v].push_back(e.size());
        e.emplace_back(u,0);
    }
    int work(int s,int t){
        int ans=0;
        while(bfs(s,t)){
            cur.assign(n+1,0);
            ans+=dfs(s,t,MAXN);
        }
        return ans;
    }
};
```

### 最小割

网络 $G=(V,E)$ 的一个割 ${S,T}$，$S$ 和 $T$ 是点的一个划分，$s\in S,t\in T$，${S,T}$ 的容量= ${\textstyle \sum_{u\in S} \sum_{v\in T}c(u,v)}$

最小割要找到一个割，使得容量尽可能小

根据最大流最小割定理，最大流=最小割，直接套用最大流即可

$dep[x]!=-1$，表示属于割边，$x$ 与 $s$ 联通

### 最小费用最大流

在网络上对每条边 $(u,v)$ 给定一个权值 $w(u,v)$，称为费用，含义是单位流量通过 $(u,v)$ 所花费的代价，对于 $G$ 所有可能的最大流中总费用最小的为最小费用最大流，SSP算法，$O(nm+n^2f)$，其中 $f$ 为网络最大流

```c++
struct MinCostFlow{
    const int MAXN=1e18;
    struct edge{
        int y,f,c;
        edge(int y,int f,int c):y(y),f(f),c(c){}
    };
    const int n;
    vector<edge> e;
    vector<vector<int>> g;
    vector<int> h,dis;
    vector<int> pre;
    void spfa(int s,int t){
        queue<int> q;
        vector<bool> vis(n+1,0);
        h[s]=0,vis[s]=1;
        q.push(s);
        while(!q.empty()){
            int u=q.front();
            q.pop();
            vis[u]=0;
            for(int i:g[u]){
                const auto &[y,f,c]=e[i];
                if(f&&h[y]>h[u]+c){
                    h[y]=h[u]+c;
                    if(!vis[y]){
                        vis[y]=1;
                        q.push(y);
                    }
                }
            }
        }
    }
    bool dijkstra(int s,int t){
        dis.assign(n+1,MAXN);
        pre.assign(n+1,-1);
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<>> q;
        dis[s]=0;
        q.emplace(0,s);
        while(!q.empty()){
            auto [D,x]=q.top();
            q.pop();
            if(dis[x]<D) continue;
            for(int i:g[x]){
                const auto &[y,f,c]=e[i];
                if(f&&dis[y]>D+h[x]-h[y]+c){
                    dis[y]=D+h[x]-h[y]+c;
                    pre[y]=i;
                    q.emplace(dis[y],y);
                }
            }
        }
        return dis[t]!=MAXN;
    }
    MinCostFlow(int n):n(n),g(n+1){}
    //x->y f:流量 c:费用
    void add(int x,int y,int f,int c){
        g[x].push_back(e.size());
        e.emplace_back(y,f,c);
        g[y].push_back(e.size());
        e.emplace_back(x,0,-c);
    }
    pair<int,int> work(int s,int t){
        int flow=0;
        int cost=0;
        h.assign(n+1,MAXN);
        spfa(s,t);
        while(dijkstra(s,t)){
            for(int i=0;i<=n;i++) h[i]+=dis[i];
            int aug=MAXN;
            for(int i=t;i!=s;i=e[pre[i]^1].y){
                aug=min(aug,e[pre[i]].f);
            }
            for(int i=t;i!=s;i=e[pre[i]^1].y){
                e[pre[i]].f-=aug;
                e[pre[i]^1].f+=aug;
            }
            flow+=aug;
            cost+=aug*h[t];
        }
        return make_pair(flow,cost);
    }
};
```

zkw

```cpp
struct MinCostFlow {
    const int INF = 1e18;
    struct edge {
        int y, f, c;
        edge(int y, int f, int c) :y(y), f(f), c(c) {}
    };
    const int n;
    vector<edge> e;
    vector<vector<int>> g;
    vector<int> h, dis, pre;
    vector<int> it;
    vector<char> vis;

    MinCostFlow(int n) :n(n), g(n + 1) {}
    //x->y，流量f，费用c
    void add(int x, int y, int f, int c) {
        g[x].push_back(e.size());
        e.emplace_back(y, f, c);
        g[y].push_back(e.size());
        e.emplace_back(x, 0, -c);
    }
    void spfa(int s) {
        h.assign(n + 1, INF);
        deque<int> q;
        vector<bool> vis(n + 1, 0);
        h[s] = 0;
        q.push_back(s);
        vis[s] = 1;
        while (!q.empty()) {
            int u = q.front();
            q.pop_front();
            vis[u] = 0;
            for (int id : g[u]) {
                const auto& ed = e[id];
                if (ed.f == 0) continue;
                int v = ed.y;
                int nd = h[u] + ed.c;
                if (nd < h[v]) {
                    h[v] = nd;
                    if (!vis[v]) {
                        if (!q.empty() && nd < h[q.front()]) q.push_front(v);
                        else q.push_back(v);
                        vis[v] = 1;
                    }
                }
            }
        }
    }
    bool dijkstra(int s, int t) {
        dis.assign(n + 1, INF);
        pre.assign(n + 1, -1);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        dis[s] = 0; pq.emplace(0, s);
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d != dis[u]) continue;
            for (int id : g[u]) {
                const auto& ed = e[id];
                if (ed.f == 0) continue;
                int v = ed.y;
                int nd = d + ed.c + h[u] - h[v];
                if (nd < dis[v]) {
                    dis[v] = nd; pre[v] = id;
                    pq.emplace(nd, v);
                }
            }
        }
        return dis[t] != INF;
    }
    int dfs(int u, int t, int f) {
        if (u == t || !f) return f;
        vis[u] = 1;
        int used = 0;
        for (int& k = it[u]; k < (int)g[u].size(); ++k) {
            int id = g[u][k], v = e[id].y;
            if (!e[id].f || vis[v]) continue;
            if (e[id].c + h[u] - h[v] != 0) continue;
            int pushed = dfs(v, t, min(f, e[id].f));
            if (pushed) {
                e[id].f -= pushed;
                e[id ^ 1].f += pushed;
                used += pushed;
                f -= pushed;
                if (!f) break;
            }
        }
        return used;
    }

    pair<int, int> work(int s, int t) {
        spfa(s);
        int flow = 0, cost = 0;
        while (dijkstra(s, t)) {
            for (int i = 0;i <= n;i++) if (dis[i] != INF) h[i] += dis[i];
            if (it.size() != n + 1) it.assign(n + 1, 0);
            else fill(it.begin(), it.end(), 0);
            if (vis.size() != n + 1) vis.assign(n + 1, 0);
            const int unit_cost = h[t] - h[s];
            int aug;
            do {
                fill(vis.begin(), vis.end(), 0);
                aug = dfs(s, t, INF);
                if (aug) {
                    flow += aug;
                    cost += aug * unit_cost;
                }
            } while (aug);
        }
        return { flow,cost };
    }
};
```

### 跑完最大流后求最大独立集和最小点覆盖方案

1. 在残量图中沿着正向边流量未满或者反向边有流量的边跑dfs，找到可达点。
2. 最小点覆盖包括
   - 左边不可达的点
   - 右边可达的点
3. 最大独立集即是最小点覆盖的补
   - 左边可达的点
   - 右边不可达的点

## 差分约束

n元一次不等式组，包含n个变量x1……xn，以及m个约束条件，形如xi-xj<=ck，其中ck为常量。令dis0等于0，0向所有的点连一条点权为0的边，dis[i]<=dis[j]+ck，则j到i连一条长度为ck的边。如果存在负环则无解。

## 强连通分量

### tarjan

```cpp
vector<int> dfn(n + 1), low(n + 1), belong(n + 1);
    vector<bool> instack(n + 1);
    stack<int> st;
    vector<vector<int>> ans;
    int cnt = 0;
    auto tarjan = [&](auto &&self, int x) -> void {
        dfn[x] = low[x] = ++cnt;
        st.push(x);
        instack[x] = 1;
        for (int &p : g[x]) {
            if (!dfn[p]) {
                self(self, p);
                low[x] = min(low[x], low[p]);
            } else if (instack[p]) {
                low[x] = min(low[x], dfn[p]);
            }
        }
        if (low[x] == dfn[x]) {
            ans.push_back({});
            while (!st.empty() && st.top() != x) {
                int f = st.top();
                st.pop();
                ans.back().push_back(f);
                belong[f] = (int)ans.size() - 1;
                instack[f] = 0;
            }
            st.pop();
            ans.back().push_back(x);
            instack[x] = 0;
            belong[x] = (int)ans.size() - 1;
        }
    };
    for (int i = 1; i <= n; i++) {
        if (!dfn[i])
            tarjan(tarjan, i);
    }
```

### Kosaraju 科萨拉朱

两次dfs，第一次dfs原图，第二次dfs反图，第二次每次dfs到的点属于同一个强连通块

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<pair<int,int>>> v(n+1);
    vector<vector<pair<int,int>>> vv(n+1);
    vector<pair<int,int>> e(1);
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        e.emplace_back(x,y);
        v[x].emplace_back(y,i);
        vv[y].emplace_back(x,i);
    }
    vector<bool> vis1(n+1),vis2(n+1);
    vector<int> s;
    vector<bool> usee(m+1);
    auto dfs1=[&](auto &&self,int x)->void{
        for(auto &[to,id]:v[x]){
            if(vis1[to]) continue;
            usee[id]=1;
            vis1[to]=1;
            self(self,to);
        }
        s.push_back(x);
    };
    auto dfs2=[&](auto &&self,int x)->void{
        for(auto &[to,id]:vv[x]){
            if(vis2[to]) continue;
            usee[id]=1;
            vis2[to]=1;
            self(self,to);
        }
    };
    for(int i=1;i<=n;i++){
        if(!vis1[i]){
            vis1[i]=1;
            dfs1(dfs1,i);
        }
    }
    int _=0;
    for(int i=n-1;i>=0;i--){
        if(!vis2[s[i]]){
            _=1;
            vis2[s[i]]=1;
            dfs2(dfs2,s[i]);
        }
    }
    int cnt=0;
    for(int i=1;i<=m&&cnt<m-2*n;i++){
        if(!usee[i]){
            cout<<e[i].first<<" "<<e[i].second<<"\n";
            cnt++;
        }
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    cin>>t;
    while(t--) solve();
    return 0;
}
```

## 割点与桥

如果某个顶点u，存在一个子节点v使得lowv>=dfnu，不能回到祖先，则u为割点，根节点需要单独考虑，如果遍历了一个子节点就可以将所有点都遍历完，那根节点就不是割点，否则是割点

```cpp
    int cnt=0;
    vector<int> dfn(n+1,0),low(n+1,0);
    vector<bool> flag(n+1,0);
    function<void(int,int)> tarjan=[&](int x,int fa){
        int son=0;
        low[x]=dfn[x]=++cnt;
        for(int &p:v[x]){
            if(!dfn[p]){
                son++;
                tarjan(p,x);
                low[x]=min(low[x],low[p]);
                if(low[p]>=dfn[x]){
                    flag[x]=1;
                }
            }else if(p!=fa){
                low[x]=min(low[x],dfn[p]);
            }
        }
        if(!fa&&son<=1){
            flag[x]=0;
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]){
            tarjan(i,0);
        }
    }
```

如果某个顶点u，存在一个子节点v使得lowv>dfnu，则u-v是割边，不用特判根节点

flag[x]=1，表示fa[x]->x是桥

```cpp
    int cnt=0;
    vector<int> dfn(n+1,0),low(n+1,0);
    vector<bool> flag(n+1,0);
    function<void(int,int)> tarjan=[&](int x,int fa){
        int son=0;
        low[x]=dfn[x]=++cnt;
        for(int &p:v[x]){
            if(!dfn[p]){
                son++;
                tarjan(p,x);
                low[x]=min(low[x],low[p]);
                if(low[p]>dfn[x]){
                    flag[p]=1;
                }
            }else if(p!=fa){
                low[x]=min(low[x],dfn[p]);
            }
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]){
            tarjan(i,0);
        }
    }
```

## 双连通分量

### 边双连通分量

先求出桥，把桥删去，剩下的极大连通子图就是边双连通分量，不能用常规方法存图

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> v(n+1);
    vector<int> e;
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        v[x].push_back(e.size());
        e.push_back(y);
        v[y].push_back(e.size());
        e.push_back(x);
    }
    int cnt=0;
    vector<int> dfn(n+1,0),low(n+1,0);
    vector<bool> flag(e.size(),0);
    function<void(int,int)> tarjan=[&](int x,int laste){
        low[x]=dfn[x]=++cnt;
        for(int &p:v[x]){
            int to=e[p];
            if(p==(laste^1)) continue;
            if(!dfn[to]){
                tarjan(to,p);
                low[x]=min(low[x],low[to]);
                if(low[to]>dfn[x]){
                    flag[p]=flag[p^1]=1;
                }
            }else{
                low[x]=min(low[x],dfn[to]);
            }
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]){
            tarjan(i,2*m);
        }
    }
    vector<bool> vis(n+1,0);
    vector<vector<int>> ebcc;
    function<void(int,int)> dfs=[&](int x,int id){
        ebcc[id].push_back(x);
        vis[x]=1;
        for(int &p:v[x]){
            if(vis[e[p]]||flag[p]) continue;
            dfs(e[p],id);
        }
    };
    for(int i=1;i<=n;i++){
        if(!vis[i]){
            ebcc.push_back({});
            dfs(i,ebcc.size()-1);
        }
    }
    cout<<ebcc.size()<<"\n";
    for(auto &p:ebcc){
        cout<<p.size()<<" ";
        for(auto &q:p){
            cout<<q<<" ";
        }
        cout<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> v(n+1);
    vector<int> e;
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        v[x].push_back(e.size());
        e.push_back(y);
        v[y].push_back(e.size());
        e.push_back(x);
    }
    int cnt=0;
    vector<int> dfn(n+1,0),low(n+1,0);
    vector<bool> flag(e.size(),0);
    stack<int> st;
    vector<vector<int>> ebcc;
    function<void(int,int)> tarjan=[&](int x,int laste){
        low[x]=dfn[x]=++cnt;
        st.push(x);
        for(int &p:v[x]){
            int to=e[p];
            if(p==(laste^1)) continue;
            if(!dfn[to]){
                tarjan(to,p);
                low[x]=min(low[x],low[to]);
            }else{
                low[x]=min(low[x],dfn[to]);
            }
        }
        if(dfn[x]==low[x]){
            ebcc.push_back({});
            while(!st.empty()&&st.top()!=x){
                ebcc.back().push_back(st.top());
                st.pop();
            }
            ebcc.back().push_back(x);
            st.pop();
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]){
            tarjan(i,2*m);
        }
    }
    cout<<ebcc.size()<<"\n";
    for(auto &p:ebcc){
        cout<<p.size()<<" ";
        for(auto &q:p){
            cout<<q<<" ";
        }
        cout<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

- 建图时需要考虑**重边**（一定不是桥）；建图后 `ebcc.work()` 求解；

```cpp
struct EBCC {
    int n, m;
    vector<vector<pair<int, int>>> g;  // {邻居, 边索引}
    vector<int> dfn, low;
    vector<bool> bridge;
    int cur;  // current time

    int cnt;                   // number of ebccs
    vector<int> comp;          // component id of node i (0-indexed)
    vector<vector<int>> node;  // nodes in each ebcc (0-indexed)

    EBCC(int _n) : n(_n), g(_n) {
        m = 0;  // number of edges
    }

    // 添加无向边，并赋予其唯一索引
    void add(int u, int v) {
        g[u].emplace_back(v, m);
        g[v].emplace_back(u, m);
        m++;
    }

    // Tarjan DFS，通过父边索引来识别桥
    void dfs1(int u, int faidx) {
        dfn[u] = low[u] = ++cur;
        for (const auto& edge : g[u]) {
            int v = edge.first;
            int curidx = edge.second;
            if (curidx == faidx) {
                continue;
            }

            if (!dfn[v]) {
                dfs1(v, curidx);
                low[u] = std::min(low[u], low[v]);
                if (low[v] > dfn[u]) {
                    bridge[curidx] = true;
                }
            } else {
                low[u] = std::min(low[u], dfn[v]);
            }
        }
    }

    // 常规DFS，不经过桥，划分E-BCC
    void dfs2(int u, int id) {
        comp[u] = id;
        node[id].push_back(u);  // 0-indexed

        for (const auto& edge : g[u]) {
            int v = edge.first;
            int curidx = edge.second;
            if (bridge[curidx] || comp[v] != -1) {
                continue;
            }
            dfs2(v, id);
        }
    }

    void work() {
        dfn.assign(n, 0);
        low.assign(n, 0);
        bridge.assign(m, false);
        cur = cnt = 0;
        comp.assign(n, -1);
        node.clear();

        for (int i = 0; i < n; i++) {
            if (!dfn[i]) dfs1(i, -1);
        }

        for (int i = 0; i < n; i++) {
            if (comp[i] == -1) {
                node.emplace_back();
                dfs2(i, cnt);
                cnt++;
            }
        }
    }
};
```

### 点双连通分量

两个点双最多有一个公共点，且一定是割点。

对于一个点双，他在dfs搜索树中dfn值最小的一定是割点或者树根

当一个点是割点的时候，他一定是点双的根

当一个点是树根的时候：

如果有两个及以上的子树，他是割点

只有一个子树，他是一个点双的根

没有子树，他自己是一个点双

需要特判自环的情况

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> v(n+1);
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        if(x!=y){
            v[x].push_back(y);
            v[y].push_back(x);
        }
    }
    int cnt=0;
    vector<int> dfn(n+1,0),low(n+1,0);
    stack<int> st;
    vector<vector<int>> pbcc;
    function<void(int,int)> tarjan=[&](int x,int root){
        low[x]=dfn[x]=++cnt;
        st.push(x);
        if(x==root&&v[x].size()==0){
            pbcc.push_back({});
            pbcc.back().push_back(x);
            return;
        }
        for(int &p:v[x]){
            if(!dfn[p]){
                tarjan(p,root);
                low[x]=min(low[x],low[p]);
                if(low[p]>=dfn[x]){
                    pbcc.push_back({});
                    while(!st.empty()&&st.top()!=p){
                        pbcc.back().push_back(st.top());
                        st.pop();
                    }
                    pbcc.back().push_back(p);
                    st.pop();
                    pbcc.back().push_back(x);
                }
            }else{
                low[x]=min(low[x],dfn[p]);
            }
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]){
            tarjan(i,i);
        }
    }
    cout<<pbcc.size()<<"\n";
    for(auto &p:pbcc){
        cout<<p.size()<<" ";
        for(auto &q:p){
            cout<<q<<" ";
        }
        cout<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

```cpp
//自环会死！！！！！
#include<bits/stdc++.h>
using namespace std;
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<pair<int,int>>> v(n+1);
    vector<pair<int,int>> e(m+1);
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        if(x==y) continue;
        e[i]={x,y};
        v[x].emplace_back(y,i);
        v[y].emplace_back(x,i);
    }
    int cnt=0;
    vector<int> dfn(n+1,0),low(n+1,0);
    stack<int> st;
    vector<vector<int>> pbcc;
    vector<vector<int>> pbccedge;
    function<void(int,int,int)> tarjan=[&](int x,int root,int pid){
        low[x]=dfn[x]=++cnt;
        if(x==root&&v[x].size()==0){
            pbcc.push_back({});
            pbcc.back().push_back(x);
            pbccedge.emplace_back();
            return;
        }
        for(auto &[p,id]:v[x]){
            if(id==pid) continue;
            if(!dfn[p]){
                st.push(id);
                tarjan(p,root,id);
                low[x]=min(low[x],low[p]);
                if(low[p]>=dfn[x]){
                    pbccedge.emplace_back();
                    while(st.top()!=id){
                        pbccedge.back().push_back(st.top());
                        st.pop();
                    }
                    pbccedge.back().push_back(id);
                    st.pop();
                    pbcc.emplace_back();
                    for(auto &t:pbccedge.back()){
                        pbcc.back().push_back(e[t].first);
                        pbcc.back().push_back(e[t].second);
                    }
                    sort(pbcc.back().begin(),pbcc.back().end());
                    pbcc.back().erase(unique(pbcc.back().begin(),pbcc.back().end()),pbcc.back().end());
                }
            }else if(dfn[p]<dfn[x]){
                st.push(id);
                low[x]=min(low[x],dfn[p]);
            }
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]){
            tarjan(i,i,-1);
        }
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    cin>>t;
    while(t--) solve();
    return 0;
}
```

- 外部建图无需考虑重边和自环，传入 `VBCC` 后得到V-BCC（根据题目要求处理孤立点）

```cpp
struct VBCC {
    int n;
    const vector<vector<int>>& g;
    vector<int> dfn, low;
    stack<int> stk;
    int cur;  // current time

    int cnt;                   // number of vbccs
    vector<vector<int>> node;  // nodes in each vbcc
    vector<bool> cut;          // 割点

    VBCC(const vector<vector<int>>& _g) : n(_g.size()), g(_g) {
        dfn.assign(n, 0);
        low.assign(n, 0);
        cut.assign(n, false);
        while (!stk.empty()) stk.pop();
        node.clear();
        cur = cnt = 0;

        for (int i = 0; i < n; ++i) {
            if (!dfn[i]) dfs(i, -1);
        }

        // 如果题目要求孤立点也算一个V-BCC
        vector<bool> vis(n, false);
        for (const auto& c : node) {
            for (int x : c) {
                vis[x] = true;
            }
        }

        for (int i = 0; i < n; i++) {
            if (!vis[i]) {
                cnt++;
                node.push_back({i});
            }
        }
    }

    void dfs(int u, int fa) {
        dfn[u] = low[u] = ++cur;
        stk.push(u);
        int child = 0;  // number of children in DFS tree
        if (g[u].empty() && fa == -1) return;
        for (int v : g[u]) {
            if (v == fa) continue;
            if (!dfn[v]) {
                child++;
                dfs(v, u);
                low[u] = min(low[u], low[v]);
                if (low[v] >= dfn[u]) {
                    if (fa != -1) cut[u] = true;
                    cnt++;
                    node.emplace_back();
                    while (true) {
                        int t = stk.top();
                        stk.pop();
                        node.back().push_back(t);
                        if (t == v) break;
                    }
                    node.back().push_back(u);  // 割点
                }
            } else {
                low[u] = min(low[u], dfn[v]);
            }
        }
        if (fa == -1 && child > 1) {
            cut[u] = true;
        }
    }
};
```

直接处理出每个点双的子图

```cpp
struct VBCC {
    int n;
    const vector<vector<int>> &g;
    vector<int> dfn, low;
    stack<pair<int, int>> st;
    int cur;

    int cnt;
    vector<vector<int>> node;                // 点集
    vector<vector<pair<int, int>>> subgraph; // 子图
    vector<bool> cut;

    VBCC(const vector<vector<int>> &_g) : n(_g.size()), g(_g) {
        dfn.assign(n, 0);
        low.assign(n, 0);
        cut.assign(n, false);
        while (!st.empty())
            st.pop();
        subgraph.clear();
        node.clear();
        cur = cnt = 0;

        for (int i = 1; i < n; ++i) {
            if (!dfn[i])
                dfs(i, -1);
        }
        vector<bool> vis(n, false);
        for (const auto &c : node) {
            for (int x : c) {
                vis[x] = true;
            }
        }
        for (int i = 1; i < n; i++) {
            if (!vis[i]) {
                cnt++;
                node.push_back({i});
                subgraph.push_back({});
            }
        }
    }

    void dfs(int u, int fa) {
        dfn[u] = low[u] = ++cur;
        int child = 0;
        if (g[u].empty() && fa == -1)
            return;
        for (int v : g[u]) {
            if (v == fa)
                continue;
            if (!dfn[v]) {
                child++;
                st.push({u, v});
                dfs(v, u);
                low[u] = min(low[u], low[v]);
                if (low[v] >= dfn[u]) {
                    if (fa != -1)
                        cut[u] = true;
                    cnt++;
                    node.emplace_back();
                    subgraph.emplace_back();
                    auto &comp_nodes = node.back();
                    auto &comp_edges = subgraph.back();
                    while (true) {
                        auto e = st.top();
                        st.pop();
                        comp_edges.push_back(e);
                        if ((e.first == u && e.second == v) || (e.first == v && e.second == u)) {
                            break;
                        }
                    }
                    set<int> points;
                    for (auto &edge : comp_edges) {
                        points.insert(edge.first);
                        points.insert(edge.second);
                    }
                    points.insert(u);
                    comp_nodes.assign(points.begin(), points.end());
                }
            } else {
                if (dfn[v] < dfn[u]) {
                    st.push({u, v});
                }
                low[u] = min(low[u], dfn[v]);
            }
        }
        if (fa == -1 && child > 1) {
            cut[u] = true;
        }
    }
};
```

## 最短路

### SPFA

```cpp
queue<int> q;
vector<bool> vis(n+1,0);
vector<int> dis(n+1,1e18);
dis[0]=0;
q.push(0);
vis[0]=1;
while(!q.empty()){
    int f=q.front();
    q.pop();
    vis[f]=0;
    for(auto &[to,w]:v[f]){
        if(dis[to]>dis[f]+w){
            dis[to]=dis[f]+w;
            if(!vis[to]){
                vis[to]=1;
                q.push(to);
            }
        }
    }
}
```

### dijkstra

时间复杂度O(mlogm)

```cpp
priority_queue<pair<int,int>,vector<pair<int,int>>,greater<>> q;
vector<bool> vis(n+1,0);
vector<int> dis(n+1,INF);
dis[1]=0;
q.push({0,1});
while(!q.empty()){
    auto [d,x]=q.top();
    q.pop();
    if(vis[x]) continue;
    vis[x]=1;
    for(auto &[to,w]:v[x]){
        if(dis[to]>dis[x]+w){
            dis[to]=dis[x]+w;
            q.push({dis[to],to});
        }
    }
}
```

### Johnson

建立虚点0，向每个点连一条边权位0的有向边。先进行一次SPFA，求出虚点0到每个点i的最短路$h[i]$，将每条边$v[i][j]$的边权设置为$v[i][j]+h[i]-h[j]$，边权变为非负数，跑dijkstra，到k的最短路为$dis[k]+h[k]-h[s]$

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int INF=1e9;
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<pair<int,int>>> v(n+1);
    for(int i=1;i<=m;i++){
        int x,y,k;
        cin>>x>>y>>k;
        v[x].push_back({y,k});
    }
    queue<int> q;
    vector<int> h(n+1,INF);
    vector<bool> vis(n+1,0);
    vector<int> cnt(n+1,0);
    q.push(0);
    vis[0]=1;
    h[0]=0;
    bool ok=1;
    while(!q.empty()){
        int f=q.front();
        q.pop();
        vis[f]=0;
        if(++cnt[f]>n+1){
            ok=0;
            break;
        }
        if(f==0){
            for(int i=1;i<=n;i++){
                if(h[i]>h[f]){
                    h[i]=h[f];
                    if(!vis[i]){
                        vis[i]=1;
                        q.push(i);
                    }
                }
            }
        }else{
            for(auto &[to,w]:v[f]){
                if(h[to]>h[f]+w){
                    h[to]=h[f]+w;
                    if(!vis[to]){
                        vis[to]=1;
                        q.push(to);
                    }
                }
            }
        }
    }
    if(!ok){
        cout<<-1<<"\n";
        return;
    }
    for(int i=1;i<=n;i++){
        for(auto &[to,w]:v[i]){
            w=w+h[i]-h[to];
        }
    }
    for(int i=1;i<=n;i++){
        fill(vis.begin(),vis.end(),0);
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<>> q;
        vector<int> dis(n+1,INF);
        dis[i]=0;
        q.push({0,i});
        while(!q.empty()){
            auto [d,x]=q.top();
            q.pop();
            if(vis[x]) continue;
            vis[x]=1;
            for(auto &[to,w]:v[x]){
                if(dis[to]>dis[x]+w){
                    dis[to]=dis[x]+w;
                    q.push({dis[to],to});
                }
            }
        }
        int sum=0;
        for(int j=1;j<=n;j++){
            if(dis[j]==INF) sum+=j*INF;
            else sum+=j*(dis[j]+h[j]-h[i]);
        }
        cout<<sum<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 普通环计数

n个点m条边无向图，求简单环数量

$dp[i][j]$状压，表示i的状态下，从i的__buildtin_ctz点作为起点，有多少种情况。i表示经过的点，j表示现在在的点

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long 
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> dp(1ll<<20,vector<int>(n));
    vector<vector<int>> v(n);
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        --x,--y;
        v[x].push_back(y);
        v[y].push_back(x);
    }
    for(int i=0;i<n;i++){
        dp[1ll<<i][i]=1;
    }
    int ans=0;
    for(int i=1;i<(1ll<<n);i++){
        for(int j=0;j<n;j++){
            if(!dp[i][j]) continue;
            for(int &p:v[j]){
                if((i&-i)>(1ll<<p)) continue;
                if(i&(1ll<<p)){
                    if((i&-i)==(1ll<<p)){
                        ans+=dp[i][j];
                    }
                }else{
                    dp[i|(1ll<<p)][p]+=dp[i][j];
                }
            }
        }
    }
    cout<<(ans-m)/2<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 三元环计数

给所有边定向，从度数小的指向度数大的，度数相同的从编号小的指向编号大的，此时图变成有向无环图DAG。枚举u和u指向的点v，再枚举v指向的点w，检验u，w是否相连，时间复杂度$O(m\sqrt{m})$

给一个n个点m条边的简单无向图，求三元环个数

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    int n,m;
    cin>>n>>m;
    vector<pair<int,int>> v(m);
    vector<int> cnt(n+1,0);
    vector<set<int>> son(n+1);
    for(int i=0;i<m;i++){
        cin>>v[i].first>>v[i].second;
        cnt[v[i].first]++;
        cnt[v[i].second]++;
    }
    for(int i=0;i<m;i++){
        if(cnt[v[i].first]==cnt[v[i].second]){
            if(v[i].first>v[i].second) swap(v[i].first,v[i].second);
        }else{
            if(cnt[v[i].first]>cnt[v[i].second]){
                swap(v[i].first,v[i].second);
            }
        }
    }
    vector<vector<int>> node(n+1);
    for(int i=0;i<m;i++){
        node[v[i].first].push_back(v[i].second);
    }
    int ans=0;
    vector<bool> vis(n+1,0);
    for(int i=1;i<=n;i++){
        for(int &p:node[i]) vis[p]=1;
        for(int &p:node[i]){
            for(int &q:node[p]){
                if(vis[q]) ans++;
            }
        }
        for(int &p:node[i]) vis[p]=0;
    }
    cout<<ans<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 2-SAT问题

n个集合，每个集合有两个元素，已知若干个<a,b>，表示a与b矛盾（a，b不属于同一个集合），需要从每个集合中选择一个元素，判断能否选n个两两不矛盾元素。

a1和b2有矛盾，则建有向边a1->b1,b2->a2，tarjan缩点判断是否有一个集合中的两个元素都在同一个强连通块，如果是则不可能。

选择的时候，优先选择dfs序大的，即scc编号小的

也可以爆搜

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> v(2*n+1);
    while(m--){
        int i,a,j,b;
        cin>>i>>a>>j>>b;
        if(a==0){
            if(b==0){
                v[i+n].push_back(j);
                v[j+n].push_back(i);
            }else{
                v[i+n].push_back(j+n);
                v[j].push_back(i);
            }
        }else{
            if(b==0){
                v[i].push_back(j);
                v[j+n].push_back(i+n);
            }else{
                v[i].push_back(j+n);
                v[j].push_back(i+n);
            }
        }
    }
    vector<int> low(2*n+1,0),dfn(2*n+1,0),belong(2*n+1,0);
    vector<bool> instack(2*n+1,0);
    int cnt=0;
    int id=0;
    stack<int> st;
    function<void(int)> tarjan=[&](int x){
        low[x]=dfn[x]=++cnt;
        st.push(x);
        instack[x]=1;
        for(int &p:v[x]){
            if(!dfn[p]){
                tarjan(p);
                low[x]=min(low[x],low[p]);
            }else if(instack[p]){
                low[x]=min(low[x],dfn[p]);
            }
        }
        if(low[x]==dfn[x]){
            id++;
            while(!st.empty()&&st.top()!=x){
                int f=st.top();
                st.pop();
                instack[f]=0;
                belong[f]=id;
            }
            belong[st.top()]=id;
            instack[st.top()]=0;
            st.pop();
        }
    };
    for(int i=1;i<=2*n;i++){
        if(!dfn[i]){
            tarjan(i);
        }
    }
    for(int i=1;i<=n;i++){
        if(belong[i]==belong[i+n]){
            cout<<"IMPOSSIBLE\n";
            return;
        }
    }
    cout<<"POSSIBLE\n";
    for(int i=1;i<=n;i++){
        if(belong[i]>belong[i+n]) cout<<1<<" ";
        else cout<<0<<" ";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 树链剖分

两次dfs，第一次求出fa，dep，son，sz，第二次求出dfn，top，rnk（dfs序对应的点编号）。

点权，要先init

```cpp
struct HLD{
    int n;
    vector<vector<int>> v;
    vector<int> fa,dep,son,sz,dfn,top,rnk;
    int cnt,root;
    SegmentTree tree;
    HLD(int n,int root):n(n),root(root){
        v.resize(n+1);
        fa.resize(n+1,0);
        dep.resize(n+1,0);
        son.resize(n+1,-1);
        sz.resize(n+1,1);
        dfn.resize(n+1,0);
        top.resize(n+1,0);
        rnk.resize(n+1,0);
        cnt=0;
    }
    void addedge(int x,int y){
        v[x].push_back(y);
        v[y].push_back(x);
    }
    void dfs1(int x,int father){
        dep[x]=dep[father]+1;
        fa[x]=father;
        for(int &p:v[x]){
            if(p==father) continue;
            dfs1(p,x);
            sz[x]+=sz[p];
            if(son[x]==-1||sz[p]>sz[son[x]]) son[x]=p;
        }
    }
    void dfs2(int x,int father,int u){
        dfn[x]=++cnt;
        rnk[cnt]=x;
        top[x]=u;
        if(son[x]==-1) return;
        dfs2(son[x],x,u);
        for(int &p:v[x]){
            if(p==father||p==son[x]) continue;
            dfs2(p,x,p);
        }
    }
    void init(vector<int> &v){
        dfs1(root,0);
        dfs2(root,0,root);
        vector<int> vv(n+1);
        for(int i=1;i<=n;i++){
            vv[dfn[i]]=v[i];
        }
        tree=SegmentTree(n,vv);
    }
    int LCA(int x,int y){
        while(top[x]!=top[y]){
            if(dep[top[x]]<dep[top[y]]) swap(x,y);
            x=fa[top[x]];
        }
        return dep[x]>dep[y]?y:x;
    }
    void updateroute(int x,int y,int delta){
        while(top[x]!=top[y]){
            if(dep[top[x]]<dep[top[y]]) swap(x,y);
            tree.update(1,1,n,dfn[top[x]],dfn[x],delta);
            x=fa[top[x]];
        }
        if(dep[x]>dep[y]) swap(x,y);
        tree.update(1,1,n,dfn[x],dfn[y],delta);
    }
    int queryroute(int x,int y){
        int ans=0;
        while(top[x]!=top[y]){
            if(dep[top[x]]<dep[top[y]]) swap(x,y);
            ans+=tree.query(1,1,n,dfn[top[x]],dfn[x]);
            x=fa[top[x]];
        }
        if(dep[x]>dep[y]) swap(x,y);
        return ans+tree.query(1,1,n,dfn[x],dfn[y]);
    }
    void updatesubtree(int x,int delta){
        tree.update(1,1,n,dfn[x],dfn[x]+sz[x]-1,delta);
    }
    int querysubtree(int x){
        return tree.query(1,1,n,dfn[x],dfn[x]+sz[x]-1);
    }
};
```

```cpp
int p;
struct SegmentTree{
    struct edge{
        int sum;
    };
    vector<int> lazy;
    vector<edge> node;
    int n;
    void pushup(int id,int l,int r){
        node[id].sum=(node[id<<1].sum+node[id<<1|1].sum)%p;
    }
    void pushdown(int id,int l,int r){
        if(lazy[id]){
            int mid=l+(r-l>>1);
            lazy[id<<1]+=lazy[id];
            lazy[id<<1|1]+=lazy[id];
            node[id<<1].sum+=(mid-l+1)*lazy[id]%p;
            node[id<<1|1].sum+=(r-mid)*lazy[id]%p;
            lazy[id<<1]%=p;
            lazy[id<<1|1]%=p;
            node[id<<1].sum%=p;
            node[id<<1|1].sum%=p;
            lazy[id]=0;
        }
    }
    SegmentTree(int n):n(n){
        node.resize((n<<2)+5);
        lazy.assign((n<<2)+5,0);
    }
    SegmentTree(){}
    void init(vector<int> &v){
        function<void(int,int,int)> buildtree=[&](int id,int l,int r){
            lazy[id]=0;
            if(l==r){
                node[id].sum=v[l]%p;
                return;
            }
            int mid=l+(r-l>>1);
            buildtree(id<<1,l,mid);
            buildtree(id<<1|1,mid+1,r);
            pushup(id,l,r);
        };
        buildtree(1,1,n);
    }
    SegmentTree(int n,vector<int> &v):n(n){
        node.resize((n<<2)+5);
        lazy.assign((n<<2)+5,0);
        init(v);
    }
    void update(int id,int l,int r,int x,int y,int delta){
        if(x<=l&&r<=y){
            lazy[id]+=delta;
            node[id].sum+=delta*(r-l+1);
            lazy[id]%=p;
            node[id].sum%=p;
            return;
        }
        pushdown(id,l,r);
        int mid=l+(r-l>>1);
        if(x<=mid) update(id<<1,l,mid,x,y,delta);
        if(y>mid) update(id<<1|1,mid+1,r,x,y,delta);
        pushup(id,l,r);
    }
    int query(int id,int l,int r,int x,int y){
        if(x<=l&&r<=y) return node[id].sum;
        pushdown(id,l,r);
        int mid=l+(r-l>>1);
        int ans=0;
        if(x<=mid) ans+=query(id<<1,l,mid,x,y);
        ans%=p;
        if(y>mid) ans+=query(id<<1|1,mid+1,r,x,y);
        ans%=p;
        return ans;
    }
};
struct HLD{
    int n;
    vector<vector<int>> v;
    vector<int> fa,dep,son,sz,dfn,top,rnk;
    int cnt,root;
    SegmentTree tree;
    HLD(int n,int root):n(n),root(root){
        v.resize(n+1);
        fa.resize(n+1,0);
        dep.resize(n+1,0);
        son.resize(n+1,-1);
        sz.resize(n+1,1);
        dfn.resize(n+1,0);
        top.resize(n+1,0);
        rnk.resize(n+1,0);
        cnt=0;
    }
    void addedge(int x,int y){
        v[x].push_back(y);
        v[y].push_back(x);
    }
    void dfs1(int x,int father){
        dep[x]=dep[father]+1;
        fa[x]=father;
        for(int &p:v[x]){
            if(p==father) continue;
            dfs1(p,x);
            sz[x]+=sz[p];
            if(son[x]==-1||sz[p]>sz[son[x]]) son[x]=p;
        }
    }
    void dfs2(int x,int father,int u){
        dfn[x]=++cnt;
        rnk[cnt]=x;
        top[x]=u;
        if(son[x]==-1) return;
        dfs2(son[x],x,u);
        for(int &p:v[x]){
            if(p==father||p==son[x]) continue;
            dfs2(p,x,p);
        }
    }
    void init(vector<int> &v){
        dfs1(root,0);
        dfs2(root,0,root);
        vector<int> vv(n+1);
        for(int i=1;i<=n;i++){
            vv[dfn[i]]=v[i];
        }
        tree=SegmentTree(n,vv);
    }
    int LCA(int x,int y){
        while(top[x]!=top[y]){
            if(dep[top[x]]<dep[top[y]]) swap(x,y);
            x=fa[top[x]];
        }
        return dep[x]>dep[y]?y:x;
    }
    void updateroute(int x,int y,int delta){
        delta%=p;
        while(top[x]!=top[y]){
            if(dep[top[x]]<dep[top[y]]) swap(x,y);
            tree.update(1,1,n,dfn[top[x]],dfn[x],delta);
            x=fa[top[x]];
        }
        if(dep[x]>dep[y]) swap(x,y);
        tree.update(1,1,n,dfn[x],dfn[y],delta);
    }
    int queryroute(int x,int y){
        int ans=0;
        while(top[x]!=top[y]){
            if(dep[top[x]]<dep[top[y]]) swap(x,y);
            ans+=tree.query(1,1,n,dfn[top[x]],dfn[x]);
            ans%=p;
            x=fa[top[x]];
        }
        if(dep[x]>dep[y]) swap(x,y);
        return (ans+tree.query(1,1,n,dfn[x],dfn[y]))%p;
    }
    void updatesubtree(int x,int delta){
        delta%=p;
        tree.update(1,1,n,dfn[x],dfn[x]+sz[x]-1,delta);
    }
    int querysubtree(int x){
        return tree.query(1,1,n,dfn[x],dfn[x]+sz[x]-1);
    }
};
```

## 表达式树

中序表达式转换为逆波兰式

符号优先级：^:3，*/:2，+-:1，():0

s[i]中：

①如果是数字，压入结果栈

②如果是乘方，压入符号栈

③如果是+-*/，将符号栈栈顶比他优先级高或相同的符号一一弹出，并压入结果栈，然后将s[i]压入符号栈

④如果是左括号，压入符号栈

⑤如果是右括号，将符号栈顶部第一个左括号之前的符号弹出并压入结果栈，并弹出左括号

最后将符号栈中剩余的符号都弹出并压入结果栈

## 树哈希

用于判断树是否同构

以某个节点为根的子树的哈希值，就是以它的所有儿子为根的子树的哈希值构成的多重集的哈希值

$h_x=f(\{h_i|i \in son(x) \})$

其中哈希函数为：$f(S)=(c+\sum_{x\in s}g(x)) mod~m$

其中c为常数，一般使用1。g为整数到整数的映射

m棵树，每行第一个数n表示点数，接下来n个数表示每个点的父节点。找出与每个树同构的树的最小编号。

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int MOD=1e9+7;
mt19937_64 myrnd(time(0));
int rnd(){
    return myrnd()>>1;
}
map<int,int> mp;
int gethash(int x){
    if(mp.find(x)!=mp.end()) return mp[x];
    else return mp[x]=rnd();
}
void solve(){
    int m;
    cin>>m;
    map<int,int> mp1;
    for(int i=0;i<m;i++){
        int n;
        cin>>n;
        vector<vector<int>> v(n+1);
        int root;
        for(int j=1;j<=n;j++){
            int x;
            cin>>x;
            if(x==0) root=j;
            else v[x].push_back(j);
        }
        vector<int> hash(n+1);
        function<void(int,int)> dfs1=[&](int x,int fa){
            hash[x]=1;
            for(int &p:v[x]){
                if(p==fa) continue;
                dfs1(p,x);
                hash[x]=(hash[x]+gethash(hash[p]))%MOD;
            }
        };
        dfs1(root,0);
        set<int> se;
        function<void(int,int)> dfs2=[&](int x,int fa){
            se.insert(hash[x]);
            for(int &p:v[x]){
                if(p==fa) continue;
                int a=hash[x],b=hash[p];
                hash[x]=((hash[x]-gethash(hash[p]))%MOD+MOD)%MOD;
                hash[p]=(hash[p]+gethash(hash[x]))%MOD;
                dfs2(p,x);
                hash[x]=a,hash[p]=b;
            }
        };
        dfs2(root,0);
        int ans=i+1;
        for(auto &p:se){
            if(mp1.find(p)==mp1.end()){
                mp1[p]=i+1;
            }else{
                ans=min(ans,mp1[p]);
            }
        }
        cout<<ans<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 欧拉通路&回路

Hierholzer

```cpp
auto dfs=[&](auto self,int x)->void{
        for(;pos[x]<v[x].size();){
            pos[x]++;
            self(self,v[x][pos[x]-1]);
        }
        st.push(x);
    };
```

```cpp
stack<int> st;
    auto dfs=[&](auto self,int x)->void{
        while(pos[x]<v[x].size()){
            int g=v[x][pos[x]];
            if(use[g]){
                pos[x]++;
                continue;
            }
            use[g]=1;
            use[g^1]=1;
            pos[x]++;
            self(self,e[g]);
        }
        st.push(x);
    };
    dfs(dfs,s);
```

```cpp
void solve(){
    int m;
    cin>>m;
    vector<int> e;
    vector<int> in(501,0);
    vector<int> pos(501,0);
    vector<bool> use;
    vector<vector<int>> v(501);
    for(int i=1;i<=m;i++){
        int x,y;
        cin>>x>>y;
        v[x].push_back(e.size());
        e.push_back(y);
        v[y].push_back(e.size());
        e.push_back(x);
        in[x]++;
        in[y]++;
        use.push_back(0);
        use.push_back(0);
    }
    for(int i=1;i<=500;i++){
        sort(v[i].begin(),v[i].end(),[&](int &a,int &b){
            return e[a]<e[b];
        });
    }
    vector<int> tmp,tmp1;
    for(int i=1;i<=500;i++){
        if(in[i]==0) continue;
        if(in[i]%2==0){
            tmp.push_back(i);
        }else{
            tmp1.push_back(i);
        }
    }
    sort(tmp.begin(),tmp.end());
    sort(tmp1.begin(),tmp1.end());
    int s;
    if(tmp1.size()!=0){
        s=tmp1[0];
    }else{
        s=tmp[0];
    }
    stack<int> st;
    auto dfs=[&](auto self,int x)->void{
        while(pos[x]<v[x].size()){
            if(use[v[x][pos[x]]]){
                pos[x]++;
                continue;
            }
            use[v[x][pos[x]]]=1;
            use[v[x][pos[x]]^1]=1;
            pos[x]++;
            self(self,e[v[x][pos[x]-1]]);
        }
        st.push(x);
    };
    dfs(dfs,s);
    while(!st.empty()){
        cout<<st.top()<<"\n";
        st.pop();
    }
}
```

## 拓扑排序

```cpp
bool toposort(){
    vector<int> list;
    queue<int> q;
    for (int i = 1; i <= n; i++) {
        if (in[i] == 0) q.push(i);
    }
    while(!q.empty()){
        int cur = q.front(); q.pop();
        list.push_back(cur);
        for (auto &to : g[cur]) {
            if (--in[to] == 0) q.push(to);
        }
    }
    if (list.size() == n) {
        for (auto &i : list) {
            cout << i << ' ';
        };
        cout << '\n';
        return true;
    }
    return false;
}
```

## 拓扑序计数

```cpp
vector<int> dp(1 << n);
dp[0] = 1;
for (int i = 0; i < m; i++) {
    int u, v;
    cin >> u >> v;
    g[v] |= 1 << u;
}
for (int i = 0; i < 1 << n; i++) {
    for (int j = 0; j < n; j++) {
        if ((g[j] & i) == g[j] && !(i >> j & 1)) {
            (dp[i | (1 << j)] += dp[i]) %= mod;
        }
    }
}
```

## 最小路径覆盖

最小路径覆盖为用最少的路径走过有向图所有点

最小路径覆盖=点数-拆点后二分图最大匹配

```cpp
//每行输出一条路径，最后一行输出路径条数
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct Flow{
    const int n;
    const int MAXN=1e18;
    vector<pair<int,int>> e;
    vector<vector<int>> g;
    vector<int> cur,dep;
    Flow(int n):n(n),g(n+1){}
    bool bfs(int s,int t){
        dep.assign(n+1,-1);
        queue<int> q;
        dep[s]=0;
        q.push(s);
        while(!q.empty()){
            const int u=q.front();
            q.pop();
            for(int i:g[u]){
                auto [v,c]=e[i];
                if(c>0&&dep[v]==-1){
                    dep[v]=dep[u]+1;
                    if(v==t) return 1;
                    q.push(v);
                }
            }
        }
        return 0;
    }
    int dfs(int u,int t,int f){
        if(u==t) return f;
        int res=f;
        for(int &i=cur[u];i<g[u].size();i++){
            const int j=g[u][i];
            auto [v,c]=e[j];
            if(c>0&&dep[v]==dep[u]+1){
                int out=dfs(v,t,min(res,c));
                e[j].second-=out;
                e[j^1].second+=out;
                res-=out;
                if(res==0) return f;
            }
        }
        return f-res;
    }
    void add(int u,int v,int c){
        g[u].push_back(e.size());
        e.emplace_back(v,c);
        g[v].push_back(e.size());
        e.emplace_back(u,0);
    }
    int work(int s,int t){
        int ans=0;
        while(bfs(s,t)){
            cur.assign(n+1,0);
            ans+=dfs(s,t,MAXN);
        }
        return ans;
    }
};
void solve(){
    int n,m;
    cin>>n>>m;
    Flow flow(2*n+2);
    for(int i=0;i<m;i++){
        int x,y;
        cin>>x>>y;
        flow.add(x+n,y,1);
    }
    for(int i=1;i<=n;i++){
        flow.add(2*n+1,n+i,1);
        flow.add(i,2*n+2,1);
    }
    int ans=n-flow.work(2*n+1,2*n+2);
    vector<int> in(n+1,0);
    vector<vector<int>> v(n+1);
    for(int i=1;i<=n;i++){
        for(int id:flow.g[i+n]){
            if(flow.e[id].first>n) continue;
            if(flow.e[id^1].second){
                v[i].push_back(flow.e[id].first);
                in[flow.e[id].first]++;
            }
        }
    }
    for(int i=1;i<=n;i++){
        if(in[i]) continue;
        int now=i;
        while(1){
            cout<<now<<" ";
            if(v[now].empty()) break;
            now=v[now].front();
        }
        cout<<"\n";
    }
    cout<<ans<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 树的重心

最大独立集

重心的所有子树中最大子数树节点数最少（<=n/2)

## 点分治

路径分为两种，一种经过rt，一种不经过rt

每次对于一棵树，取他的重心，求这个树里面每个点到根的距离。

继续对于他的每个儿子节点作为根的子树，继续找重心，继续上面的操作。

一定要注意getdis的时候先不要把信息合并到主树上，跑完getdis再把信息合并到已遍历的主树上。

```cpp
//树上是否存在边权和为k的路径
#include<bits/stdc++.h>
using namespace std;
void solve(){
    int n,m;
    cin>>n>>m;
    vector<vector<pair<int,int>>> v(n+1);
    for(int i=0;i<n-1;i++){
        int x,y,t;
        cin>>x>>y>>t;
        v[x].emplace_back(y,t);
        v[y].emplace_back(x,t);
    }
    vector<int> que(m),sz(n+1),dp(n+1);
    vector<bool> p1(1e7+5),ans(m),vis(n+1);
    p1[0]=1;
    for(int &p:que) cin>>p;
    function<void(int,int)> getsize=[&](int x,int fa){
        sz[x]=1;
        for(auto &[to,w]:v[x]){
            if(to==fa||vis[to]) continue;
            getsize(to,x);
            sz[x]+=sz[to];
        }
    };
    auto getzx=[&](int x,int fa,int tot){
        int rt=0;
        function<void(int,int)> dfs=[&](int x,int fa){
            int maxn=0;
            for(auto &[to,w]:v[x]){
                if(to==fa||vis[to]) continue;
                dfs(to,x);
                maxn=max(maxn,sz[to]);
            }
            maxn=max(maxn,tot-sz[x]);
            if(maxn<=tot/2) rt=x;
        };
        dfs(x,fa);
        return rt;
    };
    function<void(int)> calc=[&](int x){
        vector<int> p2;
        stack<int> st;
        function<void(int,int)> getdis=[&](int x,int fa){
            p2.push_back(dp[x]);
            for(auto &[to,w]:v[x]){
                if(to==fa||vis[to]) continue;
                dp[to]=dp[x]+w;
                getdis(to,x);
            }
        };
        for(auto &[to,w]:v[x]){
            p2.clear();
            if(vis[to]) continue;
            dp[to]=w;
            getdis(to,x);
            for(int i=0;i<m;i++){
                if(ans[i]) continue;
                for(int &p:p2){
                    if(que[i]-p>=0&&que[i]-p<=1e7&&p1[que[i]-p]){
                        ans[i]=1;
                        break;
                    }
                }
            }
            for(int &p:p2){
                if(p>1e7) continue;
                p1[p]=1;
                st.push(p);
            }
        }
        while(!st.empty()){
            p1[st.top()]=0;
            st.pop();
        }
    };
    function<void(int)> solve=[&](int x){
        vis[x]=1;
        calc(x);
        for(auto &[to,w]:v[x]){
            if(vis[to]) continue;
            getsize(to,x);
            int rt=getzx(to,x,sz[to]);
            solve(rt);
        }
    };
    getsize(1,0);
    int rt=getzx(1,0,n);
    solve(rt);
    for(int i=0;i<m;i++){
        if(ans[i]) cout<<"AYE\n";
        else cout<<"NAY\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 点分树

按照点分治的重心，构建一棵新树，也就是对于每一个找到的重心，将上一层分治的重心设置为他的父节点，得到一棵大小不变，最多$log(n)$层的虚树。

每个点子树大小的和为$nlogn$。

建虚树的时候注意别用原树的边来建了。

注意根节点是 CD.root 而不是随便一个节点

```c++
struct CD {
    int n;
    vector<vector<int>> g;
    int root;
    CD() {}
    CD(int n_, vector<vector<int>> g_) : n(n_), g(n_ + 1) {
        vector<int> sz(n + 1);
        vector<int> mx(n + 1);
        vector<int> vis(n + 1);
        mx[0] = INF;
        auto getzx = [&](auto &&self, int x, int fa, int &zx, int tot) -> void {
            sz[x] = 1;
            mx[x] = 0;
            for (auto &to : g_[x]) {
                if (to == fa || vis[to])
                    continue;
                self(self, to, x, zx, tot);
                sz[x] += sz[to];
                mx[x] = max(mx[x], sz[to]);
            }
            mx[x] = max(mx[x], tot - sz[x]);
            if (mx[x] < mx[zx])
                zx = x;
        };
        auto work = [&](auto &&self, int x, int tot) -> int {
            int zx = 0;
            getzx(getzx, x, 0, zx, tot);
            vis[zx] = 1;
            for (auto &to : g_[zx]) {
                if (vis[to])
                    continue;
                int v;
                if (sz[to] > sz[zx])
                    v = self(self, to, tot - sz[zx]);
                else
                    v = self(self, to, sz[to]);
                g[zx].emplace_back(v);
                g[v].emplace_back(zx);
            }
            return zx;
        };
        root = work(work, 1, n);
    }
};
```

## Kruskal重构树

```cpp
struct DSU{
    vector<int> fa, sz;
    int n;
    DSU(int n) : n(n){
        fa.resize(n + 1);
        sz.resize(n + 1, 1);
        for (int i = 1; i <= n; i++){
            fa[i] = i;
        }
    }
    int find(int a){
        return fa[a] == a ? a : fa[a] = find(fa[a]);
    }
    void merge(int x, int y){
        int fx = find(x), fy = find(y);
        if (fx == fy)
            return;
        if (sz[fx] < sz[fy])
            swap(fx, fy);
        fa[fy] = fx;
        sz[fx] += sz[fy];
        return;
    }
};
struct edge {
    int x, y, k;
};
struct Ex_kruskal {
    int n;
    vector<vector<int>> fa;
    vector<int> val;
    vector<int> dep;
    DSU dsu;
    int LCA(int x, int y) {
        // 不连通返回-1
        if (dsu.find(x) != dsu.find(y))
            return -1;
        if (dep[x] < dep[y])
            swap(x, y);
        for (int i = 20; i >= 0; i--) {
            if (dep[fa[x][i]] >= dep[y]) {
                x = fa[x][i];
            }
        }
          if(x==y) return x;
        for (int i = 20; i >= 0; i--) {
            if (fa[x][i] != fa[y][i]) {
                x = fa[x][i];
                y = fa[y][i];
            }
        }
        return val[fa[x][0]];
    }
    Ex_kruskal(int &_n, vector<edge> &_v) : n(_n), dsu(2 * _n), fa(2 * _n + 1, vector<int>(21)), val(2 * _n + 1), dep(2 * _n + 1) {
        sort(_v.begin(), _v.end(), [&](edge &a, edge &b) { return a.k < b.k; });
        vector<vector<int>> v(2 * _n + 1);
        vector<bool> vis(2 * _n + 1);
        int cnt = 1;
        for (int i = 0; i < _v.size(); i++) {
            int fax = dsu.find(_v[i].x);
            int fay = dsu.find(_v[i].y);
            if (fax != fay) {
                v[fax].emplace_back(n + cnt);
                v[fay].emplace_back(n + cnt);
                v[n + cnt].emplace_back(fax);
                v[n + cnt].emplace_back(fay);
                val[n + cnt] = _v[i].k;
                dsu.fa[fax] = n + cnt;
                dsu.fa[fay] = n + cnt;
                cnt++;
            }
            if (cnt == n) {
                break;
            }
        }
        auto dfs = [&](auto &&self, int x, int father) -> void {
            vis[x] = 1;
            dep[x] = dep[father] + 1;
            fa[x][0] = father;
            for (int i = 1; i <= 20; i++) {
                fa[x][i] = fa[fa[x][i - 1]][i - 1];
            }
            for (int &to : v[x]) {
                if (to == father)
                    continue;
                self(self, to, x);
            }
        };
        for (int i = 2 * n - 1; i > 0; i--) {
            if (!vis[i])
                dfs(dfs, i, 0);
        }
    }
};
```

## 最近公共祖先

### 倍增

时间复杂度$O(logn)$

```cpp
vector<int> dep(n + 1, 0);
vector<vector<int>> fa(n + 1, vector<int>(21));
auto dfs = [&](auto &&self, int x, int father) -> void
{
    dep[x] = dep[father] + 1;
    fa[x][0] = father;
    for (int i = 1; i <= 20; i++)
    {
        fa[x][i] = fa[fa[x][i - 1]][i - 1];
    }
    for (int &p : v[x])
    {
        if (p == father)
            continue;
        self(self, p, x);
    }
};
auto LCA = [&](int x, int y)
{
    if (dep[x] < dep[y])
        swap(x, y);
    for (int i = 20; i >= 0; i--)
    {
        if (dep[fa[x][i]] >= dep[y])
            x = fa[x][i];
    }
    if (x == y)
        return x;
    for (int i = 20; i >= 0; i--)
    {
        if (fa[x][i] != fa[y][i])
        {
            x = fa[x][i];
            y = fa[y][i];
        }
    }
    return fa[x][0];
};
dfs(dfs, 1, 0);
```

### tarjan

预处理$O(n+q)$，询问$O(1)$

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct DSU{
    vector<int> fa;
    int n;
    DSU(int n){
        this->n=n;
        fa=vector<int>(n+1);
        for(int i=0;i<=n;i++) fa[i]=i;
    }
    int find(int x){
        return fa[x]==x?x:fa[x]=find(fa[x]);
    }
    bool merge(int x,int y){
        int fax=find(x);
        int fay=find(y);
        if(fax==fay) return 0;
        fa[fax]=fay;
        return 1;
    }
};
void solve(){
    int n,m,s;
    cin>>n>>m>>s;
    vector<vector<int>> v(n+1);
    for(int i=1;i<n;i++){
        int x,y;
        cin>>x>>y;
        v[x].push_back(y);
        v[y].push_back(x);
    }
    DSU dsu(n);
    vector<vector<pair<int,int>>> query(n+1);
    vector<int> lca(m);
    for(int i=0;i<m;i++){
        int x,y;
        cin>>x>>y;
        query[x].emplace_back(y,i);
        query[y].emplace_back(x,i);
    }
    vector<int> dfn(n+1),low(n+1);
    stack<int> st;
    vector<bool> instack(n+1);
    int cnt=0;
    function<void(int)> tarjan=[&](int x){
        dfn[x]=low[x]=++cnt;
        st.push(x);
        instack[x]=1;
        for(int &p:v[x]){
            if(!dfn[p]){
                tarjan(p);
                low[x]=min(low[x],low[p]);
                dsu.merge(p,x);
            }else if(instack[p]){
                low[x]=min(low[x],dfn[p]);
            }
        }
        if(low[x]==dfn[x]){
            while(!st.empty()&&st.top()!=x){
                int f=st.top();
                st.pop();
                instack[f]=0;
            }
            st.pop();
            instack[x]=0;
        }
        for(auto &[y,id]:query[x]){
            if(lca[id]||!dfn[y]) continue;
            lca[id]=dsu.find(y);
        }
    };
    tarjan(s);
    for(int i=0;i<m;i++){
        cout<<lca[i]<<"\n";
    }
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

### 欧拉序+st表

预处理$O(nlogn)$，询问$O(1)$

```c++
struct ST {
    vector<vector<pair<int, int>>> dp;
    ST() {}
    ST(int n, vector<pair<int, int>> &v) {
        dp.resize(20);
        for (int i = 0; i < 20; i++) {
            dp[i].resize(n + 1);
        }
        for (int i = 1; i <= n; i++) {
            dp[0][i] = v[i];
        }
        for (int i = 1; i <= 18; i++) {
            for (int j = 1; j + (1ll << i) - 1 <= n; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j + (1ll << i - 1)]);
            }
        }
    }
    int query(int l, int r) {
        int k = __lg(r - l + 1);
        return min(dp[k][l], dp[k][r - (1ll << k) + 1]).second;
    }
};
struct LCA {
    int n;
    vector<pair<int, int>> lcaqu;
    vector<int> lcadfn, dep, dis;
    ST st;
    LCA(int _n, vector<vector<pair<int, int>>> _g) : n(_n), lcaqu(1), lcadfn(n + 1), dep(n + 1), dis(n + 1) {
        auto lcadfs = [&](auto &&self, int x, int fa) -> void {
            lcadfn[x] = lcaqu.size();
            dep[x] = dep[fa] + 1;
            lcaqu.emplace_back(dep[x], x);
            for (auto &[to, w] : _g[x]) {
                if (to == fa)
                    continue;
                dis[to] = dis[x] + w;
                self(self, to, x);
                lcaqu.emplace_back(dep[x], x);
            }
        };
        lcadfs(lcadfs, 1, 0);
        st = ST(lcaqu.size() - 1, lcaqu);
    }
    LCA(int _n, vector<vector<int>> _g) : n(_n), lcaqu(1), lcadfn(n + 1), dep(n + 1), dis(n + 1) {
        auto lcadfs = [&](auto &&self, int x, int fa) -> void {
            lcadfn[x] = lcaqu.size();
            dep[x] = dep[fa] + 1;
            lcaqu.emplace_back(dep[x], x);
            for (auto &to : _g[x]) {
                if (to == fa)
                    continue;
                dis[to] = dis[x] + 1;
                self(self, to, x);
                lcaqu.emplace_back(dep[x], x);
            }
        };
        lcadfs(lcadfs, 1, 0);
        st = ST(lcaqu.size() - 1, lcaqu);
    }
    int getlca(int x, int y) {
        int l = lcadfn[x], r = lcadfn[y];
        if (l > r)
            swap(l, r);
        return st.query(l, r);
    }
    // 求边权dis，点权不要用。
    int getdis(int x, int y) {
        int lca = getlca(x, y);
        return dis[x] + dis[y] - 2 * dis[lca];
    }
};
```

## 图的匹配

### 一般图最大匹配

- 带花树算法（Blossom Algorithm）$O(|V||E|^2)$
- 顶点索引 `0-indexed`；`g.add(u,v)` 双向边；
- 调用 `g.findMatching` 返回匹配数组，`match[u] = v` : 表示顶点 `u` 和顶点 `v` 相互匹配；`match[u] = -1`: 表示顶点 `u` 是未匹配点。可用 `std::count_if` 易得最大匹配对数。

```cpp
struct Graph {
    int n;
    vector<vector<int>> g;
    Graph(int n) : n(n), g(n) {}
    void add(int u, int v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> findMatching() {
        vector<int> match(n, -1), vis(n), link(n), f(n), dep(n);
        // disjoint set union
        auto find = [&](int u) {
            while (f[u] != u)
                u = f[u] = f[f[u]];
            return u;
        };
        auto lca = [&](int u, int v) {
            u = find(u);
            v = find(v);
            while (u != v) {
                if (dep[u] < dep[v])
                    std::swap(u, v);
                u = find(link[match[u]]);
            }
            return u;
        };
        queue<int> que;
        auto blossom = [&](int u, int v, int p) {
            while (find(u) != p) {
                link[u] = v;
                v = match[u];
                if (vis[v] == 0) {
                    vis[v] = 1;
                    que.push(v);
                }
                f[u] = f[v] = p;
                u = link[v];
            }
        };
        // find an augmenting path starting from u and augment (if exist)
        auto augment = [&](int u) {
            while (!que.empty()) que.pop();
            std::iota(f.begin(), f.end(), 0);
            // vis = 0 corresponds to inner vertices
            // vis = 1 corresponds to outer vertices
            std::fill(vis.begin(), vis.end(), -1);
            que.push(u);
            vis[u] = 1;
            dep[u] = 0;
            while (!que.empty()) {
                int u = que.front();
                que.pop();
                for (auto v : g[u]) {
                    if (vis[v] == -1) {
                        vis[v] = 0;
                        link[v] = u;
                        dep[v] = dep[u] + 1;
                        // found an augmenting path
                        if (match[v] == -1) {
                            for (int x = v, y = u, temp; y != -1; x = temp, y = x == -1 ? -1 : link[x]) {
                                temp = match[y];
                                match[x] = y;
                                match[y] = x;
                            }
                            return;
                        }
                        vis[match[v]] = 1;
                        dep[match[v]] = dep[u] + 2;
                        que.push(match[v]);
                    } else if (vis[v] == 1 && find(v) != find(u)) {
                        // found a blossom
                        int p = lca(u, v);
                        blossom(u, v, p);
                        blossom(v, u, p);
                    }
                }
            }
        };
        // find a maximal matching greedily (decrease constant)
        auto greedy = [&]() {
            for (int u = 0; u < n; ++u) {
                if (match[u] != -1) continue;
                for (auto v : g[u]) {
                    if (match[v] == -1) {
                        match[u] = v;
                        match[v] = u;
                        break;
                    }
                }
            }
        };
        greedy();
        for (int u = 0; u < n; ++u) {
            if (match[u] == -1) augment(u);
        }
        return match;
    }
};

void solve() {
    int n, m;
    cin >> n >> m;
    Graph g(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        g.add(u, v);
    }

    auto match = g.findMatching();
    int cnt = count_if(match.begin(), match.end(), [](int x) { return x != -1; }) / 2;
    cout << cnt << "\n";
    for (int i = 0; i < match.size(); i++) {
        cout << match[i] + 1 << " \n"[i == match.size() - 1];
    }
}
```

## 竞赛图

任意两个点之间有且只有一条边的有向图（有向完全图），赢的点连向输的点

### 兰道定理

比分序列：将每个点的出度从小到大排序的序列

设s是图G的比分序列，G是竞赛图的充要条件为：
$$
\sum_{i=1}^k s_i \geq C_k^2, \quad \forall k \in \{1, 2, \ldots, n\}
$$
并且在k=n的时候取等

### 竞赛图强连通分量个数

$$
\sum_{i=1}^n \left[\sum_{j=1}^i s_j = \binom{i}{2}\right]
$$

## prufer序列

长度为n-2，与一个顶点标过号的点数为n的无根树一一对应。

无根树转prufer序列：

1. 找到编号最小的度数为1的点
2. 删除该节点，并在序列中添加与该节点相连的节点的编号
3. 重复1,2操作，直到只剩下两个节点

prufer转无根树:

1. 每次取出prufer序列中最前面的元素u
2. 在点集中找到编号最小的没有在prufer中出现的元素v
3. u,v连边，分别删除
4. 最后点集剩下两个点，连边

性质:

1. prufer中某个编号出现的次数等于这个点在无根树中的度数-1
2. n点的无根树唯一对应了一个长度为n-2的数列，数列中每个数都在1到n的范围内
3. n个点的无向完全图的生成树个数：$n^{n-2}$
4. n个节点度依次为$d_1,d_2,...,d_n$的无根树共有$\frac{(n-2)!}{\prod_{i=1}^n(d_i-1)!}$个
5. n个点的有标号有根树共有 $n^{(n-2)}*n=n^{n-1}$ 个

# 计算几何

## 二维几何

### Levis

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct Point{
    double x,y;
    double operator*(const Point &e) const{
        return x*e.x+y*e.y;
    };
    Point operator*(const double k) const{
        return {x*k,y*k};
    }
    double operator^(const Point &e) const{
        return x*e.y-e.x*y;   
    }
    Point operator+(const Point &e) const{
        return {x+e.x,y+e.y};
    }
    Point operator-(const Point &e) const{
        return {x-e.x,y-e.y};
    }
    Point operator/(const double &k) const{
        return {x/k,y/k};
    }
    //象限
    inline int quad() const{
        if(x>0&&y>=0) return 1;
        if(x<=0&&y>0) return 2;
        if(x<0&&y<=0) return 3;
        if(x>=0&&y<0) return 4;
        return 5;
    }
    inline static bool sortxupyup(const Point &a,const Point &b){
        if(a.x!=b.x) return a.x<b.x;
        else return a.y<b.y;
    }
    //模长
    inline double norm() const{
        return sqrtl(x*x+y*y);
    }
    inline double norm2() const{
        return x*x+y*y;
    }
    //极角排序
    inline static bool sortPointAngle(const Point &a,const Point &b){
        if(a.quad()!=b.quad()) return a.quad()<b.quad();
        double cr=a^b;
        if(cr!=0) return cr>0;
        else return a.norm2()<b.norm2();
    }
    //向量方向
    //1 this在e顺时针方向
    //0 同向或反向
    //2 this在e逆时针方向
    int ordervector(const Point &e){
        double p=(*this)^e;
        if(p>0) return 1;
        else if(p==0.0) return 0;
        else return 2;
    }
    //逆时针旋转alpha角
    inline Point Spin(double alpha){
        double sinalpha=sin(alpha);
        double cosinalpha=cos(alpha);
        return {x*cosinalpha-y*sinalpha,x*sinalpha+y*cosinalpha};
    }
    //绕center逆时针旋转alpha角
    Point rotate(const Point &center,double alpha) const{
        return ((*this)-center).Spin(alpha)+center;
    }
    inline double dis(const Point &e){
        Point c=(*this)-e;
        return c.norm();
    }
    double getangle(const Point &e) const{
        return fabs(atan2l(*this^e,*this*e));
    }
    Point(double x,double y):x(x),y(y){}
    Point(){}
};
struct ClosestPair{
    vector<Point> p;
    vector<int> tmp_idx;
    ClosestPair(vector<Point> points) : p(points) {
        tmp_idx.resize(p.size());
    }
    void merge_sort(vector<int> &idx, int l, int r) {
        if (l >= r) return;
        int mid = (l + r) >> 1;
        int i = l, j = mid + 1, k = l;
        while (i <= mid && j <= r) {
            if (p[idx[i]].y < p[idx[j]].y) tmp_idx[k++] = idx[i++];
            else tmp_idx[k++] = idx[j++];
        }
        while (i <= mid) tmp_idx[k++] = idx[i++];
        while (j <= r) tmp_idx[k++] = idx[j++];
        for (int i = l; i <= r; i++) idx[i] = tmp_idx[i];
    }
    double solve(vector<int> &idx, int l, int r) {
        if (l >= r) return 1e18;
        if (l + 1 == r) {
            if (p[idx[l]].y > p[idx[r]].y) swap(idx[l], idx[r]);
            return p[idx[l]].dis(p[idx[r]]);
        }
        int mid = (l + r) >> 1;
        double mid_x = p[idx[mid]].x;
        double d1 = solve(idx, l, mid);
        double d2 = solve(idx, mid + 1, r);
        double d = min(d1, d2);
        merge_sort(idx, l, r);
        vector<int> strip;
        for (int i = l; i <= r; i++) {
            if (fabs(p[idx[i]].x - mid_x) < d) {
                strip.push_back(idx[i]);
            }
        }
        for (int i = 0; i < strip.size(); i++) {
            for (int j = i + 1; j < strip.size(); j++) {
                if (p[strip[j]].y - p[strip[i]].y >= d) break;
                d = min(d, p[strip[i]].dis(p[strip[j]]));
            }
        }
        return d;
    }
    //直接调用这个，返回最近两个点的dis，nlogn
    double getMinDist() {
        int n = p.size();
        if (n <= 1) return 1e18;
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return p[a].x < p[b].x || (p[a].x == p[b].x && p[a].y < p[b].y);
        });
        return solve(idx, 0, n - 1);
    }
};
struct Line{
    //过x点，方向向量为y
    Point x,y;
    //type=0,点和方向向量
    //type=1，点和点
    Line(const Point &a,const Point &b,int type){
        if(type==0){
            x=a,y=b;
        }else{
            x=a;
            y=b-a;
        }
    }
    inline double distancetopoint(const Point &e) const{
        return fabs((e-x)^y)/y.norm();
    }
    /*
    两直线关系：
    0：平行不重合
    1：有唯一交点
    2：两直线重合
    */
    inline int relation(const Line &e) const{
        if((y^e.y)!=0) return 1;
        if(((e.x-x)^y)==0) return 2;
        return 0;
    }
    /*
    求两直线交点：
    first=0：平行且不重合
    first=1：有唯一交点，second 为交点
    first=2：两直线重合
    */
    inline pair<int,Point> intersection(const Line &e) const{
        double cr=y^e.y;
        if(cr==0){
            if(((e.x-x)^y)==0){
                return {2,Point()};
            }else{
                return {0,Point()};
            }
        }
        double t=((e.x-x)^e.y)/cr;
        return {1,x+y*t};
    }
    //点e在直线上的投影
    inline Point projection(const Point &e) const{
        double t=y*(e-x)/(y*y);
        return x+y*t;
    }
    //点e关于直线的对称点
    inline Point reflection(const Point &e) const{
        Point p=projection(e);
        return p*2-e;
    }
};
struct Segment{
    //过x,y
    Point x,y;
    Segment(Point a,Point b):x(a),y(b){}
    inline double distancetopoint(const Point &e) const{
        Point l=y-x;
        double t=l*(e-x)/(l*l);
        if(t<0) t=0;
        else if(t>1) t=1;
        return (e-(x+l*t)).norm();
    }
    //判断点e是否在线段上，包含两个端点
    inline bool onsegment(const Point &e) const{
        return ((y-x)^(e-x))==0&&(e-x)*(e-y)<=0;
    }
    //两线段是否相交
    //端点接触、部分重合、完全重合都是true
    inline bool intersect(const Segment &e) const{
        double c1=(y-x)^(e.x-x);
        double c2=(y-x)^(e.y-x);
        double c3=(e.y-e.x)^(x-e.x);
        double c4=(e.y-e.x)^(y-e.x);
        if(c1==0&&onsegment(e.x)) return true;
        if(c2==0&&onsegment(e.y)) return true;
        if(c3==0&&e.onsegment(x)) return true;
        if(c4==0&&e.onsegment(y)) return true;
        return c1*c2<0&&c3*c4<0;
    }
    //两线段是否严格相交
    //必须从各自内部穿过。端点接触，共线重合都是false
    inline bool strictintersect(const Segment &e) const{
        double c1=(y-x)^(e.x-x);
        double c2=(y-x)^(e.y-x);
        double c3=(e.y-e.x)^(x-e.x);
        double c4=(e.y-e.x)^(y-e.x);
        return c1*c2<0&&c3*c4<0;
    }
    /*
    无限直线e与当前有限线段的关系：
    0：不相交
    1：在线段端点处相交
    2：严格穿过线段内部
    3：整条线段位于直线上
    */
    inline int relation(const Line &e) const{
        double c1=e.y^(x-e.x);
        double c2=e.y^(y-e.x);
        if(c1==0&&c2==0) return 3;
        if(c1==0||c2==0) return 1;
        if(c1*c2<0) return 2;
        return 0;
    }
    //两线段最短距离
    inline double distancetosegment(const Segment &e) const{
        if(intersect(e)) return 0;
        return min({distancetopoint(e.x),distancetopoint(e.y),e.distancetopoint(x),e.distancetopoint(y)});
    }
};
//要先getConvex求凸包，其他的才能用
struct Polygon{
    vector<Point> p;
    int n;
    Polygon(int n,vector<Point> &v):n(n),p(v){} 
    Polygon(int n):n(n),p(n){}
    void input(){
        for(int i=0;i<n;i++){
            cin>>p[i].x>>p[i].y;
        }
    }
    //凸包逆时针方向排序,首尾点不重复，中间共线点不保留
    void getConvex(){
        vector<Point> convexhull;
        sort(p.begin(),p.end(),Point::sortxupyup);
        p.erase(unique(p.begin(),p.end(),[](const Point &a,const Point &b){
            return a.x==b.x&&a.y==b.y;
        }),p.end());
        n=p.size();
        if(n<=2){
            return;
        }
        vector<int> st1,st2;
        for(int i=0;i<n;i++){
            while(st1.size()>=2){
                int a=st1[st1.size()-2];
                int b=st1.back();
                if(((p[b]-p[a])^(p[i]-p[b]))<=0){
                    st1.pop_back();
                }else break;
            }
            st1.push_back(i);
        }
        for(int i=n-1;i>=0;i--){
            while(st2.size()>=2){
                int a=st2[st2.size()-2];
                int b=st2.back();
                if(((p[b]-p[a])^(p[i]-p[b]))<=0){
                    st2.pop_back();
                }else break;
            }
            st2.push_back(i);
        }
        st1.pop_back();
        st2.pop_back();
        for(auto &id:st1) convexhull.push_back(p[id]);
        for(auto &id:st2) convexhull.push_back(p[id]);
        swap(convexhull,p);
        n=p.size();
    }
    double getPerimeter(){
        if(p.size()<=1) return 0;
        double ans=0;
        for(int i=0;i<p.size();i++){
            ans+=p[i].dis(p[(i+1)%p.size()]);
        }
        return ans;
    }
    double getArea(){
        if(p.size()<3) return 0;
        double ans=0;
        for(int i=0;i<p.size();i++){
            ans+=p[i]^p[(i+1)%p.size()];
        }
        return fabs(ans)/2;
    }
    //getconvex!!
    //旋转卡壳求直径，最远点对
    double getLongest(){
        const int sz=p.size();
        if(sz<=1) return 0;
        if(sz==2) return p[0].dis(p[1]);
        int j=1;
        double ans=0;
        for(int i=0;i<sz;i++){
            Line line(p[i],p[(i+1)%sz],1);
            while(line.distancetopoint(p[j])<line.distancetopoint(p[(j+1)%sz])){
                j=(j+1)%sz;
            }
            ans=max({ans,p[i].dis(p[j]),p[(i+1)%sz].dis(p[j])});
        }
        return ans;
    }
    //getConvex!!
    //旋转卡壳求凸包最小宽度
    //O(n)
    double minWidth(){
        const int sz=p.size();
        if(sz<=2) return 0;
        int j=1;
        double ans=1e18;
        for(int i=0;i<sz;i++){
            Line line(p[i],p[(i+1)%sz],1);
            while(line.distancetopoint(p[j])<
                line.distancetopoint(p[(j+1)%sz])){
                j=(j+1)%sz;
            }
            ans=min(ans,line.distancetopoint(p[j]));
        }
        return ans;
    }
    //getconvex!!
    //旋转卡壳最小矩形覆盖，返回最小面积，四个顶点（退化就是空数组）
    pair<double,vector<Point>> minRectangleCover(){
        const int sz=p.size();
        vector<Point> rec;
        if(sz<3) return {0,rec};
        int j=0,l=0,r=0;
        double ans=1e18;
        {
            Point edge=p[1]-p[0];
            Line line(p[0],p[1],1);
            for(int k=1;k<sz;k++){
                if(line.distancetopoint(p[k])>line.distancetopoint(p[j])){
                    j=k;
                }
                if(edge*(p[k]-p[0])<edge*(p[l]-p[0])){
                    l=k;
                }
                if(edge*(p[k]-p[0])>edge*(p[r]-p[0])){
                    r=k;
                }
            }
        }
        for(int i=0;i<sz;i++){
            int ni=(i+1)%sz;
            Point edge=p[ni]-p[i];
            Line line(p[i],p[ni],1);
            while(line.distancetopoint(p[j])<line.distancetopoint(p[(j+1)%sz])){
                j=(j+1)%sz;
            }
            while(edge*(p[r]-p[i])<edge*(p[(r+1)%sz]-p[i])){
                r=(r+1)%sz;
            }
            while(edge*(p[l]-p[i])>edge*(p[(l+1)%sz]-p[i])){
                l=(l+1)%sz;
            }
            double len=edge.norm();
            double height=line.distancetopoint(p[j]);
            double left=edge*(p[l]-p[i])/len;
            double right=edge*(p[r]-p[i])/len;
            double area=height*(right-left);
            if(area<ans){
                ans=area;
                Point unit=edge/len;
                Point normal={-unit.y,unit.x};
                Point p0=p[i]+unit*left;
                Point p1=p[i]+unit*right;
                rec={p0,p1,p1+normal*height,p0+normal*height};
            }
        }
        return {ans,rec};
    }
    /*
    判断点x与任意简单多边形p（首点不重复）的关系
    0：多边形外部
    1：多边形内部
    2：多边形边界
    顶点需事先按顺时针或逆时针顺序排列
    on
    */
    int pointInPolygon(const Point &x){
        int n=p.size();
        bool in=false;
        for(int i=0;i<n;i++){
            Point a=p[i];
            Point b=p[(i+1)%n];
            Segment seg(a,b);
            if(seg.onsegment(x)) return 2;
            if((a.y>x.y)!=(b.y>x.y)){
                double crossX=a.x+(b.x-a.x)*(x.y-a.y)/(b.y-a.y);
                if(crossX>x.x){
                    in=!in;
                }
            }
        }
        return in;
    }
    //getconvex！！
    //判断点是否在凸包内部或边界上，logn
    //0：外部
    //1：内部
    //2：边界
    int in(Point x){
        const int sz=p.size();
        if(sz==0) return 0;
        if(sz==1){
            return (x.x==p[0].x&&x.y==p[0].y)?2:0;
        }
        if(sz==2){
            if(((p[1]-p[0])^(x-p[0]))==0 &&
            (x-p[0])*(x-p[1])<=0){
                return 2;
            }
            return 0;
        }
        Point base=x-p[0];
        double c1=(p[1]-p[0])^base;
        double c2=(p[sz-1]-p[0])^base;
        if(c1<0 || c2>0) return 0;
        if(c1==0){
            return (x-p[0])*(x-p[1])<=0 ? 2:0;
        }
        if(c2==0){
            return (x-p[0])*(x-p[sz-1])<=0 ? 2:0;
        }
        int l=1,r=sz-1;
        while(r-l>1){
            int mid=(l+r)>>1;
            if(((p[mid]-p[0])^base)>=0){
                l=mid;
            }else{
                r=mid;
            }
        }
        double cr=(p[l+1]-p[l])^(x-p[l]);

        if(cr<0) return 0;
        if(cr==0) return 2;
        return 1;
    }
    //求简单多边形重心
    //顶点需按顺时针或逆时针排列，首点不重复
    //多边形不能退化为面积为 0
    Point polygonCentroid(){
        int n=p.size();
        double area2=0;
        Point ans(0,0);
        for(int i=0;i<n;i++){
            int j=(i+1)%n;
            double cr=p[i]^p[j];
            area2+=cr;
            ans=ans+(p[i]+p[j])*cr;
        }
        return ans/(3*area2);
    }
    //返回(整数)简单多边形内部整点，边界整点
    pair<int,int> Pick(){
        int n=p.size();
        int area2=0;
        int boundary=0;
        for(int i=0;i<n;i++){
            int j=(i+1)%n;
            int x1=llround(p[i].x);
            int y1=llround(p[i].y);
            int x2=llround(p[j].x);
            int y2=llround(p[j].y);
            area2+=x1*y2-x2*y1;
            boundary+=__gcd(abs(x1-x2),abs(y1-y2));
        }
        area2=abs(area2);
        int inside=(area2-boundary+2)/2;
        return {inside,boundary};
    }
    //用有向直线e裁剪当前多边形
    //保留e左侧和直线上的部分
    //. . .保留
    //----->
    //主要用于凸多边形，结果仍为凸多边形
    //on
    Polygon cut(const Line &e) const{
        vector<Point> res;
        int sz=p.size();
        if(sz==0) return Polygon(0);
        for(int i=0;i<sz;i++){
            Point a=p[i];
            Point b=p[(i+1)%sz];
            double ca=e.y^(a-e.x);
            double cb=e.y^(b-e.x);
            if(ca>=0){
                res.push_back(a);
            }
            if((ca<0&&cb>0)||(ca>0&&cb<0)){
                Line line(a,b,1);
                res.push_back(e.intersection(line).second);
            }
        }
        return Polygon(res.size(),res);
    }
};
struct Circle{
    //圆心，半径
    Point o;
    double r;
    Circle(Point o,double r):o(o),r(r){}
    // 由三个不共线点构造外接圆
    Circle(const Point &a,const Point &b,const Point &c){
        Point ab=b-a;
        Point ac=c-a;
        double d=2*(ab^ac);
        o=a+Point{(ab.norm2()*ac.y-ac.norm2()*ab.y)/d,(ab.x*ac.norm2()-ac.x*ab.norm2())/d};
        r=o.dis(a);
    }
    //由两点构造以ab为直径的圆
    Circle(const Point &a,const Point &b){
        o=(a+b)/2;
        r=o.dis(a);
    }
    //判断点p与圆的位置关系：
    //0：圆外
    //1：圆上
    //2：圆内
    inline int relation(const Point &p){
        double d=o.dis(p);
        if(d>r) return 0;
        if(d==r) return 1;
        return 2;
    }
    //判断直线e与圆的位置关系：
    //0：相离
    //1：相切
    //2：相交，有两个交点
    inline int relation(const Line &e){
        double d=e.distancetopoint(o);
        if(d>r) return 0;
        if(d==r) return 1;
        return 2;
    }
    //求直线e与圆的交点
    //返回数组大小为0、1、2，分别表示相离、相切、相交
    inline vector<Point> intersection(const Line &e){
        Point mid=e.projection(o);
        double d=o.dis(mid);
        if(d>r) return {};
        if(d==r) return {mid};
        double len=sqrtl(r*r-d*d);
        Point dir=e.y/e.y.norm();
        return {mid-dir*len,mid+dir*len};
    }
    //判断两圆的位置关系：
    //0：外离
    //1：外切
    //2：相交，有两个交点
    //3：内切
    //4：内含
    //5：重合
    inline int relation(const Circle &e){
        double d=o.dis(e.o);
        if(d==0&&r==e.r) return 5;
        if(d>r+e.r) return 0;
        if(d==r+e.r) return 1;
        if(d<fabs(r-e.r)) return 4;
        if(d==fabs(r-e.r)) return 3;
        return 2;
    }
    //求两圆交点
    inline vector<Point> intersection(const Circle &e){
        int type=relation(e);
        if(type==0||type==4||type==5) return {};
        double d=o.dis(e.o);
        double a=(r*r-e.r*e.r+d*d)/(2*d);
        Point dir=(e.o-o)/d;
        Point mid=o+dir*a;
        if(type==1||type==3) return {mid};
        double h=sqrtl(r*r-a*a);
        Point vertical={-dir.y,dir.x};
        return {mid+vertical*h,mid-vertical*h};
    }
    //两圆相交部分面积
    inline double intersectionArea(const Circle &e){
        double d=o.dis(e.o);
        const double PI=acosl(-1.0);
        if(d>=r+e.r) return 0;
        if(d<=fabs(r-e.r)){
            double R=min(r,e.r);
            return PI*R*R;
        }
        double cosa=(r*r+d*d-e.r*e.r)/(2*r*d);
        double cosb=(e.r*e.r+d*d-r*r)/(2*e.r*d);
        cosa=max(-1.0,min(1.0,cosa));
        cosb=max(-1.0,min(1.0,cosb));
        double a=2*acosl(cosa);
        double b=2*acosl(cosb);
        return (r*r*(a-sin(a))+e.r*e.r*(b-sin(b)))/2;
    }
    //求点p到圆的切点
    //圆外返回2个，圆上返回自身，圆内返回空
    inline vector<Point> tangentPoint(const Point &p){
        Point v=p-o;
        double d=v.norm();
        if(d<r) return {};
        if(d==r) return {p};
        double alpha=acosl(r/d);
        Point base=v/d*r;
        return {o+base.Spin(alpha),o+base.Spin(-alpha)};
    }
    //求两圆公切线
    //返回数量可能为[0,4]
    inline vector<Line> commontangent(const Circle &e){
        vector<Line> ans;
        Point d=e.o-o;
        double d2=d.norm2();
        if(d2==0) return ans;
        for(int s:{-1,1}){
            double sr=e.r*s;
            double dr=r-sr;
            double h2=d2-dr*dr;
            if(h2<0) continue;
            double h=sqrtl(h2);
            Point vertical={-d.y,d.x};
            for(int t:{-1,1}){
                Point v=(d*dr+vertical*(h*t))/d2;
                Point tangentPoint=o+v*r;
                Point tangentDirection={-v.y,v.x};
                ans.emplace_back(tangentPoint,tangentDirection,0);
                if(h2==0) break;
            }
        }
        return ans;
    }
};
Polygon Minkowski(Polygon a,Polygon b){
    if(a.p.empty()||b.p.empty()){
        return Polygon(0);
    }
    auto init=[](vector<Point> &h){
        int k=min_element(h.begin(),h.end(),[](const Point &a,const Point &b){
            if(a.y!=b.y) return a.y<b.y;
            return a.x<b.x;
        })-h.begin();
        rotate(h.begin(),h.begin()+k,h.end());
    };
    init(a.p);
    init(b.p);
    auto getEdge=[](const vector<Point> &h){
        vector<Point> edge;
        int sz=h.size();
        if(sz==1) return edge;
        for(int i=0;i<sz;i++){
            edge.push_back(h[(i+1)%sz]-h[i]);
        }
        return edge;
    };
    vector<Point> ea=getEdge(a.p);
    vector<Point> eb=getEdge(b.p);
    vector<Point> res={a.p[0]+b.p[0]};
    int i=0,j=0;
    while(i<ea.size()||j<eb.size()){
        Point edge;
        if(i<ea.size()&&j<eb.size()&&
           (ea[i]^eb[j])==0&&ea[i]*eb[j]>0){
            edge=ea[i]+eb[j];
            i++;
            j++;
        }else if(j==eb.size()||(i<ea.size()&&Point::sortPointAngle(ea[i],eb[j]))){
            edge=ea[i++];
        }else{
            edge=eb[j++];
        }
        res.push_back(res.back()+edge);
    }
    if(res.size()>1) res.pop_back();
    Polygon c(res.size(),res);
    return c;
}
/*
半平面交
每条有向直线都保留左侧及直线本身
要求最终交集是有界区域
返回结果逆时针排列，首尾不重复
空集或无界区域返回空Polygon
nlogn
*/
Polygon halfPlaneIntersection(vector<Line> lines){
    auto outside=[](const Line &e,const Point &p){
        return (e.y^(p-e.x))<0;
    };
    sort(lines.begin(),lines.end(),[](const Line &a,const Line &b){
        return Point::sortPointAngle(a.y,b.y);
    });
    vector<Line> h;
    for(auto &e:lines){
        if(h.empty()||(h.back().y^e.y)!=0){
            h.push_back(e);
        }else if((h.back().y^(e.x-h.back().x))>0){
            h.back()=e;
        }
    }
    deque<Line> q;
    deque<Point> intersectionPoint;
    for(auto &e:h){
        while(!intersectionPoint.empty()&&outside(e,intersectionPoint.back())){
            intersectionPoint.pop_back();
            q.pop_back();
        }
        while(!intersectionPoint.empty()&&outside(e,intersectionPoint.front())){
            intersectionPoint.pop_front();
            q.pop_front();
        }
        if(!q.empty()){
            auto now=q.back().intersection(e);
            if(now.first!=1) return Polygon(0);
            intersectionPoint.push_back(now.second);
        }
        q.push_back(e);
    }
    while(!intersectionPoint.empty()&&outside(q.front(),intersectionPoint.back())){
        intersectionPoint.pop_back();
        q.pop_back();
    }
    while(!intersectionPoint.empty()&&outside(q.back(),intersectionPoint.front())){
        intersectionPoint.pop_front();
        q.pop_front();
    }
    if(q.size()<3) return Polygon(0);
    auto last=q.back().intersection(q.front());
    if(last.first!=1) return Polygon(0);
    vector<Point> res(intersectionPoint.begin(),intersectionPoint.end());
    res.push_back(last.second);
    return Polygon(res.size(),res);
}
/*
最小包围圆
返回覆盖所有点的最小圆
随机增量，期望O(n)
空点集返回圆心(0,0)、半径0
*/
Circle minimumEnclosingCircle(vector<Point> p){
    if(p.empty()) return Circle(Point(0,0),0);
    mt19937_64 rng(time(0));
    shuffle(p.begin(),p.end(),rng);
    //三点确定最小圆
    //不共线时为外接圆，共线时取最远两点为直径
    auto circleFromThree=[](const Point &a,const Point &b,const Point &c){
        if(((b-a)^(c-a))!=0){
            return Circle(a,b,c);
        }
        double ab=(a-b).norm2();
        double ac=(a-c).norm2();
        double bc=(b-c).norm2();
        if(ab>=ac&&ab>=bc) return Circle(a,b);
        if(ac>=ab&&ac>=bc) return Circle(a,c);
        return Circle(b,c);
    };
    Circle ans(p[0],0);
    for(int i=1;i<p.size();i++){
        if(ans.relation(p[i])!=0) continue;
        ans=Circle(p[i],0);
        for(int j=0;j<i;j++){
            if(ans.relation(p[j])!=0) continue;
            ans=Circle(p[i],p[j]);
            for(int k=0;k<j;k++){
                if(ans.relation(p[k])!=0) continue;
                ans=circleFromThree(p[i],p[j],p[k]);
            }
        }
    }
    return ans;
}
/*
求圆c与简单多边形poly的相交面积
多边形顶点需按顺时针或逆时针排列，首点不重复
O(n)
*/
double circlePolygonIntersectionArea(const Circle &c,const Polygon &poly){
    auto calc=[&](Point a,Point b){
        a=a-c.o;
        b=b-c.o;
        Point d=b-a;
        double A=d*d;
        if(A==0) return 0.0;
        double B=2*(a*d);
        double C=a*a-c.r*c.r;
        double delta=B*B-4*A*C;
        vector<double> t={0,1};
        if(delta>0){
            double sq=sqrtl(delta);
            double t1=(-B-sq)/(2*A);
            double t2=(-B+sq)/(2*A);
            if(t1>0&&t1<1) t.push_back(t1);
            if(t2>0&&t2<1) t.push_back(t2);
        }
        sort(t.begin(),t.end());
        double ans=0;
        for(int i=0;i+1<t.size();i++){
            double l=t[i];
            double r=t[i+1];
            Point u=a+d*l;
            Point v=a+d*r;
            Point mid=a+d*((l+r)/2);
            if(mid.norm2()<=c.r*c.r){
                ans+=(u^v)/2;
            }else{
                double angle=atan2l(u^v,u*v);
                ans+=c.r*c.r*angle/2;
            }
        }
        return ans;
    };
    double ans=0;
    int n=poly.p.size();
    for(int i=0;i<n;i++){
        ans+=calc(poly.p[i],poly.p[(i+1)%n]);
    }
    return fabs(ans);
}
/*
多个圆覆盖次数面积
res[k]=至少被k个圆覆盖面积
res[1]就是圆面积并
O(n^2logn)
*/
vector<double> circleCoverageArea(vector<Circle> &circle){
    const double PI=acosl(-1);
    int n=circle.size();
    vector<double> res(n+2);
    auto normAngle=[&](double x){
        while(x<0) x+=2*PI;
        while(x>=2*PI) x-=2*PI;
        return x;
    };
    auto arcArea=[](const Circle &c,double l,double r){
        Point a=c.o+Point(cosl(l),sinl(l))*c.r;
        Point b=c.o+Point(cosl(r),sinl(r))*c.r;
        return ((c.o^(b-a))+c.r*c.r*(r-l))/2;
    };
    for(int i=0;i<n;i++){
        Circle &a=circle[i];
        vector<pair<double,int>> event;
        int cnt=1;
        for(int j=0;j<n;j++){
            if(i==j) continue;
            const Circle &b=circle[j];
            double d=a.o.dis(b.o);
            if(d==0&&a.r==b.r){
                if(j<i) continue;
                cnt++;
                continue;
            }
            if(d+a.r<=b.r){
                cnt++;
                continue;
            }
            if(d+b.r<=a.r||d>=a.r+b.r)
                continue;
            double base=atan2l(b.o.y-a.o.y,b.o.x-a.o.x);
            double cosa=(a.r*a.r+d*d-b.r*b.r)/(2*a.r*d);
            cosa=max(-1.0,min(1.0,cosa));
            double delta=acosl(cosa);
            double l=normAngle(base-delta);
            double r=normAngle(base+delta);
            if(l<=r){
                event.push_back({l,1});
                event.push_back({r,-1});
            }else{
                cnt++;
                event.push_back({r,-1});
                event.push_back({l,1});
            }
        }
        sort(event.begin(),event.end());
        double last=0;
        for(int j=0;j<event.size();){
            double now=event[j].first;
            if(last<now)
                res[cnt]+=arcArea(a,last,now);
            int k=j;
            while(k<event.size()&&event[k].first==now){
                cnt+=event[k].second;
                k++;
            }
            last=now;
            j=k;
        }
        if(last<2*PI) res[cnt]+=arcArea(a,last,2*PI);
    }
    for(int i=1;i<=n;i++) res[i]=fabs(res[i]);
    return res;
}
void solve(){

}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

### Trilliverse

#### 点线凸包

⚠默认`Point`的输入用的是`double`，如果题目给的是输入`int`请手动处理输入，具体为用`int`接受输入再调用`Polygon.add()`

```cpp
// by trilliverse
const double EPS = 1e-8;
const double PI = acos(-1.0);

inline int sgn(double x) {
    if (fabs(x) < EPS) return 0;
    return x < 0 ? -1 : 1;
}

struct Point {
    double x, y;

    Point(double _x = 0, double _y = 0) : x(_x), y(_y) {}

    // operators overloading
    Point operator+(const Point& p) const { return {x + p.x, y + p.y}; }
    Point operator-(const Point& p) const { return {x - p.x, y - p.y}; }
    Point operator*(double k) const { return {x * k, y * k}; }
    Point operator/(double k) const { return {x / k, y / k}; }

    bool operator==(const Point& p) const { return sgn(x - p.x) == 0 && sgn(y - p.y) == 0; }
    bool operator!=(const Point& p) const { return !(*this == p); }
    bool operator<(const Point& p) const {
        if (sgn(x - p.x) != 0) return x < p.x;
        return y < p.y;
    }

    double dot(const Point& p) const { return x * p.x + y * p.y; }
    double cross(const Point& p) const { return x * p.y - y * p.x; }

    double len2() const { return x * x + y * y; }
    double len() const { return hypot(x, y); }
    double dis2(const Point& p) const { return (*this - p).len2(); }
    double dis(const Point& p) const { return hypot(x - p.x, y - p.y); }

    Point rotleft() const { return {-y, x}; }   // 逆时针旋转90°
    Point rotright() const { return {y, -x}; }  // 顺时针旋转90°

    // rotate around center 'p' by 'rad' (CCW)
    Point rotate(const Point& p, double rad) const {
        Point v = (*this) - p;
        double c = cos(rad);
        double s = sin(rad);
        return {p.x + v.x * c - v.y * s, p.y + v.x * s + v.y * c};
    }

    // angle (radians or degrees) ∠APB
    // return (-π, π]: 最小有向角 正逆负顺
    double angle(const Point& a, const Point& b, bool f = true) {
        Point pa = a - (*this);
        Point pb = b - (*this);
        double ang = atan2(pa.cross(pb), pa.dot(pb));
        return f ? ang : ang * 180.0 / PI;
    }
    // return [0, 2π): p->a 到 p->b 逆时针旋转角
    double angleCCW(const Point& a, const Point& b, bool f = true) {
        double ang = angle(a, b);
        if (ang < 0) ang += 2 * PI;
        return f ? ang : ang * 180.0 / PI;
    }

    // normalize to length k
    Point norm(double k = 1.0) const {
        double L = len();
        if (sgn(L) == 0) return *this;  // 零向量
        return (*this) * (k / L);
    }

    // polar quadrant (1..4), origin -> 0
    int quad() const {
        if (sgn(x) > 0 && sgn(y) >= 0) return 1;
        if (sgn(x) <= 0 && sgn(y) > 0) return 2;
        if (sgn(x) < 0 && sgn(y) <= 0) return 3;
        if (sgn(x) >= 0 && sgn(y) < 0) return 4;
        return 0;  // 原点
    }

    // IO
    friend istream& operator>>(istream& is, Point& p) {
        return is >> p.x >> p.y;
    }
    friend ostream& operator<<(ostream& os, const Point& p) {
        os << fixed << setprecision(3) << p.x << " " << p.y;
        return os;
    }
};

// three-point helpers
inline double cross(const Point& a, const Point& b, const Point& c) {
    // cross(a,b,c) = ab × ac
    return (b - a).cross(c - a);
}
inline double dot(const Point& a, const Point& b, const Point& c) {
    // dot(a,b,c) = ab · ac
    return (b - a).dot(c - a);
}

struct Line {
    Point s, e;

    Line() {}
    Line(const Point& _s, const Point& _e) : s(_s), e(_e) {}

    // 构造: ax + by + c = 0
    Line(double a, double b, double c) {
        assert(!(sgn(a) == 0 && sgn(b) == 0) && "Invalid line coefficients");
        if (sgn(a) == 0) {  // y = -c/b

            s = {0, -c / b};
            e = {1, -c / b};
        } else if (sgn(b) == 0) {  // x = -c/a
            s = {-c / a, 0};
            e = {-c / a, 1};
        } else {
            s = {0, -c / b};
            e = {-c / a, 0};
        }
    }

    Point dir() const { return e - s; }      // 方向向量
    double len() const { return s.dis(e); }  // 线段长度

    // 直线倾斜角 [0, π)
    double angle() {
        double theta = atan2(e.y - s.y, e.x - s.x);
        if (sgn(theta) < 0) theta += PI;
        if (sgn(theta - PI) == 0) theta = 0;
        return theta;
    }

    // 点与直线/线段关系
    // 1:左侧 2:右侧 3:线段上 4:前方 5:后方
    int relationToPoint(const Point& p) const {
        int c = sgn(dir().cross(p - s));
        if (c > 0) return 1;
        if (c < 0) return 2;
        // On the line return 3
        if (sgn((p - s).dot(p - e)) <= 0) return 3;
        if (sgn((p - s).dot(dir())) > 0) return 4;
        return 5;
    }

    bool onSegment(const Point& p) const { return relationToPoint(p) == 3; }

    // 直线与直线: 1平行 2重合 3正交 4斜交
    int relationToLine(const Line& l) const {
        Point d1 = dir(), d2 = l.dir();
        double det = d1.cross(d2);
        // 叉乘为0，平行或重合
        if (sgn(det) == 0) return l.onSegment(s) ? 2 : 1;
        if (sgn(d1.dot(d2)) == 0) return 3;
        return 4;
    }

    // 两直线交点 (非平行)
    Point intersection(const Line& l) const {
        // assert(relationToLine(l) > 2);  // 保证相交
        Point d1 = dir(), d2 = l.dir();
        double det = d1.cross(d2);
        assert(sgn(det) != 0 && "Lines are parallel or coincident");
        Point diff = l.s - s;
        double t = (diff.cross(d2)) / det;
        return s + d1 * t;
    }

    // 线段与线段: 0不相交 1非规范相交 2规范相交
    int segRelation(const Line& l) const {
        int d1 = sgn(dir().cross(l.s - s));
        int d2 = sgn(dir().cross(l.e - s));
        int d3 = sgn(l.dir().cross(s - l.s));
        int d4 = sgn(l.dir().cross(e - l.s));
        if ((d1 ^ d2) == -2 && (d3 ^ d4) == -2) return 2;  // 规范相交
        bool touch = (d1 == 0 && sgn((l.s - s).dot(l.s - e)) <= 0) ||
                     (d2 == 0 && sgn((l.e - s).dot(l.e - e)) <= 0) ||
                     (d3 == 0 && sgn((s - l.s).dot(s - l.e)) <= 0) ||
                     (d4 == 0 && sgn((e - l.s).dot(e - l.e)) <= 0);
        return touch ? 1 : 0;
    }

    // 直线与线段: 0不相交 1端点相交 2规范相交
    int lineSegRelation(const Line& l) const {
        int d1 = sgn(dir().cross(l.s - s));
        int d2 = sgn(dir().cross(l.e - s));
        if ((d1 ^ d2) == -2) return 2;  // 规范相交
        return (d1 == 0 || d2 == 0);
    }

    // 点到直线距离
    double disToLine(const Point& p) const {
        return fabs((e - s).cross(p - s)) / len();
    }
    // 点到线段距离
    double disToSeg(const Point& p) const {
        Point d = dir();
        if (sgn(d.dot(p - s)) < 0 || sgn(d.dot(p - e)) > 0) {
            return min(p.dis(s), p.dis(e));  // 点到端点的距离
        }
        return disToLine(p);
    }

    // 点 p 在直线上的投影
    Point project(const Point& p) const {
        Point d = dir();
        return s + d * ((d.dot(p - s)) / d.len2());
    }

    // 点 p 关于直线的对称点
    Point reflect(const Point& p) const {
        return project(p) * 2 - p;
    }

    friend istream& operator>>(istream& is, Line& l) {
        return is >> l.s >> l.e;
    }
    friend ostream& operator<<(ostream& os, const Line& l) {
        return os << l.s << " " << l.e;
    }
};

struct Polygon {
    vector<Point> p;
    bool convex = true;  // whether to ensure convex

    Polygon() = default;
    Polygon(int n) : p(n), convex(false) {}
    Polygon(vector<Point> pt, bool f = true) : p(std::move(pt)), convex(f) {}

    inline int size() const { return (int)p.size(); }
    inline void add(const Point& q) {
        p.push_back(q);
        convex = false;
    }

    // 周长
    double perimeter() const {
        double sum = 0;
        int n = size();
        for (int i = 0; i < n; i++) sum += p[i].dis(p[(i + 1) % n]);
        return sum;
    }
    // 面积
    double area() {
        double sum = 0;
        int n = size();
        for (int i = 0; i < n; i++) sum += (p[i].cross(p[(i + 1) % n]));
        return fabs(sum) / 2;
    }
    // 质心 UVA 10002
    Point centroid() {
        int n = size();
        if (n == 0) return {0, 0};
        if (n == 1) return p[0];
        if (n == 2) return (p[0] + p[1]) / 2.0;

        Point c{0, 0};
        double area = 0;

        for (int i = 1; i < n - 1; i++) {
            double tmp = cross(p[0], p[i], p[i + 1]);
            if (sgn(tmp) == 0) continue;
            area += tmp;
            c.x += (p[0].x + p[i].x + p[i + 1].x) / 3 * tmp;
            c.y += (p[0].y + p[i].y + p[i + 1].y) / 3 * tmp;
        }
        if (sgn(area) == 0) {
            for (auto& pt : p) c = c + pt;
            return c / n;
        }
        c = c / area;
        return c;
    }

    // 多边形与点关系: 0外部 1内部 2边上 3点上
    int relationToPoint(const Point& q) const {
        int n = size();
        for (int i = 0; i < n; i++) {
            if (p[i] == q)
                return 3;
        }
        vector<Line> edge(n);
        for (int i = 0; i < n; i++) edge[i] = Line(p[i], p[(i + 1) % n]);

        for (int i = 0; i < n; i++) {
            if (edge[i].onSegment(q))
                return 2;
        }
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            int k = sgn(cross(p[j], q, p[i]));
            int u = sgn(p[i].y - q.y);
            int v = sgn(p[j].y - q.y);
            if (k > 0 && u < 0 && v >= 0)
                cnt++;
            if (k < 0 && v < 0 && u >= 0)
                cnt--;
        }
        return cnt != 0;
    }

    // 凸性判定（非严格，允许共线）
    static bool isConvex(const vector<Point>& P) {
        int n = (int)P.size();
        if (n < 3) return false;
        int pos = 0, neg = 0;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n, k = (j + 1) % n;
            int s = sgn(cross(P[i], P[j], P[k]));
            if (s > 0)
                pos++;
            else if (s < 0)
                neg++;
            if (pos && neg) return false;
        }
        return true;
    }
    bool isConvex() const { return isConvex(p); }

    // 极角排序（绕原点）
    // 若需绕某点 c，先整体平移 p[i]-=c，排完再加回
    void polarSort() {
        sort(p.begin(), p.end(), [](const Point& a, const Point& b) {
            int qa = a.quad(), qb = b.quad();
            if (qa != qb) return qa < qb;
            double cr = a.cross(b);
            if (sgn(cr) != 0) return cr > 0;
            return a.len2() < b.len2();
        });
    }

    // Andrew 求凸包（点集 -> CCW hull）
    // strict=true: 边上点不保留；false: 边上点保留
    static vector<Point> convexHull(vector<Point> pt, bool strict = true) {
        sort(pt.begin(), pt.end());
        pt.erase(unique(pt.begin(), pt.end()), pt.end());
        int n = (int)pt.size();
        if (n <= 1) return pt;

        auto collinear = [](const vector<Point>& q) {
            if (q.size() <= 2) return true;
            for (int i = 2; i < (int)q.size(); i++) {
                if (sgn(cross(q[0], q[1], q[i])) != 0) return false;
            }
            return true;
        };
        if (collinear(pt)) {
            if (strict) return {pt.front(), pt.back()};
            return pt;
        }

        vector<Point> L, U;
        // lower
        for (int i = 0; i < n; i++) {
            while ((int)L.size() >= 2) {
                double cr = cross(L[(int)L.size() - 2], L.back(), pt[i]);
                if (sgn(cr) > 0 || (!strict && sgn(cr) == 0)) break;
                L.pop_back();
            }
            L.push_back(pt[i]);
        }
        // upper
        for (int i = n - 1; i >= 0; i--) {
            while ((int)U.size() >= 2) {
                double cr = cross(U[(int)U.size() - 2], U.back(), pt[i]);
                if (sgn(cr) > 0 || (!strict && sgn(cr) == 0)) break;
                U.pop_back();
            }
            U.push_back(pt[i]);
        }
        L.pop_back();
        U.pop_back();
        L.insert(L.end(), U.begin(), U.end());
        return L;  // CCW, 无重复端点
    }

    // 若未标注凸，则原地替换为凸包
    void getConvex(bool strict = true) {
        if (convex) return;
        p = convexHull(std::move(p), strict);
        convex = true;
    }

    // 旋转卡壳求凸包直径（最大点对距离）
    double diameter() {
        getConvex();
        int n = (int)p.size();
        if (n <= 1) return 0.0;
        if (n == 2) return p[0].dis(p[1]);

        double ans2 = 0.0;
        int j = 1;
        for (int i = 0; i < n; i++) {
            int k = (i + 1) % n;
            while (fabs(cross(p[i], p[k], p[(j + 1) % n])) >
                   fabs(cross(p[i], p[k], p[j]))) {
                j = (j + 1) % n;
            }
            ans2 = max(ans2, p[i].dis2(p[j]));
            ans2 = max(ans2, p[k].dis2(p[j]));
        }
        return sqrt(ans2);
    }

    // 旋转卡壳求凸包最小矩形覆盖 UVA10173
    double minRectangleCover() {
        getConvex();
        int n = (int)p.size();
        if (n <= 2) return 0.0;

        double ans = -1;
        int j = 1, r = 1, l;
        for (int i = 0; i < n; i++) {
            // 卡出离边p[i]~p[i+1]最远点
            while (sgn(cross(p[i], p[(i + 1) % n], p[(j + 1) % n]) -
                       cross(p[i], p[(i + 1) % n], p[j])) >= 0) {
                j = (j + 1) % n;
            }
            // 卡出边p[i]~p[i+1]方向上正向最远点
            while (sgn(dot(p[i], p[(i + 1) % n], p[(r + 1) % n]) -
                       dot(p[i], p[(i + 1) % n], p[r])) >= 0) {
                r = (r + 1) % n;
            }

            if (i == 0) l = r;
            // 卡出边p[i]~p[i+1]方向上负向最远点
            while (sgn(dot(p[i], p[(i + 1) % n], p[(l + 1) % n]) -
                       dot(p[i], p[(i + 1) % n], p[l])) <= 0) {
                l = (l + 1) % n;
            }
            double d = (p[i] - p[(i + 1) % n]).len2();
            double tmp = cross(p[i], p[(i + 1) % n], p[j]) * (dot(p[i], p[(i + 1) % n], p[r]) - dot(p[i], p[(i + 1) % n], p[l])) / d;
            if (ans < 0 || ans > tmp) ans = tmp;
        }
        return ans;
    }

    // 求凸包最小内角
    double minAngle(bool rad = true) {
        getConvex();
        int n = (int)p.size();
        if (n <= 2) return 0.0;
        double minn = 200.0;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            int k = (i - 1 + n) % n;
            double angle = p[i].angleCCW(p[j], p[k]);  // ∠jik
            minn = min(minn, angle);
        }
        return rad ? minn : minn * 180.0 / PI;
    }

    friend istream& operator>>(istream& is, Polygon& poly) {
        for (auto& pt : poly.p) is >> pt;
        return is;
    }
    friend ostream& operator<<(ostream& os, const Polygon& poly) {
        for (const auto& pt : poly.p) os << pt << "\n";
        return os;
    }
};
```

#### 圆

```cpp
struct Circle {
    Point c;
    double r;
    Circle() : c(Point()), r(0) {}
    Circle(const Point& _c, double _r) : c(_c), r(_r) {}

    // 三角形外接圆 UVA12304
    Circle(Point a, Point b, Point c) {
        Line u = Line((a + b) / 2, ((a + b) / 2) + ((b - a).rotleft()));
        Line v = Line((b + c) / 2, ((b + c) / 2) + ((c - b).rotleft()));
        c = u.intersection(v);
        r = c.dis(a);
    }

    // 三角形内切圆 UVA12304
    Circle(Point a, Point b, Point c) {
        double A = b.dis(c);
        double B = c.dis(a);
        double C = a.dis(b);
        Point o = (a * A + b * B + c * C) / (A + B + C);
        c = o;
        r = fabs(cross(a, b, c)) / (0.5 * A + 0.5 * B + 0.5 * C);
    }

    double area() const { return PI * r * r; }
    double circumference() const { return 2 * PI * r; }

    // 点与圆关系: 0圆外 1圆上 2圆内
    int relationToPoint(const Point& p) const {
        double d2 = c.dis2(p);
        if (sgn(d2 - r * r) < 0)
            return 2;
        else if (sgn(d2 - r * r) == 0)
            return 1;
        else
            return 0;
    }
    // 直线与圆关系: 0相离 1相切 2相交
    int relationToLine(const Line& l) const {
        double d = l.disToLine(c);
        if (sgn(d - r) < 0)
            return 2;
        else if (sgn(d - r) == 0)
            return 1;
        else
            return 0;
    }
    // 线段与圆关系: 0相离 1相切 2相交 3线段端点在圆上 4线段端点在圆内
    int relationToSeg(const Line& l) const {
        double d1 = l.s.dis2(c);
        double d2 = l.e.dis2(c);
        if (sgn(d1 - r * r) < 0 && sgn(d2 - r * r) < 0) return 4;
        if (sgn(d1 - r * r) == 0 && sgn(d2 - r * r) == 0) return 3;
        if (sgn(d1 - r * r) == 0 || sgn(d2 - r * r) == 0) return 3;

        double d = l.disToLine(c);
        if (sgn(d - r) < 0) {
            Point p = l.project(c);
            if (l.onSegment(p))
                return 2;
            else
                return 0;
        } else if (sgn(d - r) == 0) {
            Point p = l.project(c);
            if (l.onSegment(p))
                return 1;
            else
                return 0;
        } else {
            return 0;
        }
    }

    // 圆与圆关系: 0外离 1外切 2相交 3内切 4内含
    int relationToCircle(const Circle& cir) const {
        double d = c.dis(cir.c);
        if (sgn(d - r - cir.r) > 0) return 0;
        if (sgn(d - r - cir.r) == 0) return 1;
        if (sgn(d - fabs(r - cir.r)) > 0) return 2;
        if (sgn(d - fabs(r - cir.r)) == 0) return 3;
        return 4;
    }
    // 切线
    int tangentline(const Point& p, Line& l1, Line& l2) const {
        double d2 = c.dis2(p);
        if (sgn(d2 - r * r) < 0) return 0;          // 点在圆内
        if (sgn(d2) == 0 && sgn(r) == 0) return 1;  // 点在圆心且半径为0
        if (sgn(d2 - r * r) == 0) {                 // 点在圆上
            l1 = Line(p, p + (c - p).rotleft());
            return 1;
        }
        // 点在圆外
        double d = sqrt(d2);
        double l = r * r / d2;
        double h = r * sqrt(d2 - r * r) / d2;
        Point m = c + (p - c).norm(l);
        Point v = (p - c).rotleft().norm(h);
        l1 = Line(p, m + v);
        l2 = Line(p, m - v);
        return 2;
    }
    // 圆与直线交点
    int intersectionToLine(const Line &l, Point &p1, Point &p2){
        double d = l.disToLine(c);
        if (sgn(d - r) > 0)
            return 0;
        Point proj = l.project(c);
        if (sgn(d - r) == 0){
            p1 = p2 = proj;
            return 1;
        }
        double len = sqrt(max(0.0, r * r - d * d));
        Point dir = l.dir().norm(1.0);
        p1 = proj + dir * len;
        p2 = proj - dir * len;
        return 2;
    }
    friend istream& operator>>(istream& is, Circle& cir) {
        return is >> cir.c >> cir.r;
    }
    friend ostream& operator<<(ostream& os, const Circle& cir) {
        return os << cir.c << " " << fixed << setprecision(3) << cir.r;
    }
};
```

#### 半平面交(未实现)

```cpp
struct halfplane : public Line {
    double angle;
    halfplane() {}
    //`表示向量u->v逆时针(左侧)的半平面`
    halfplane(Point _u, Point _v) {
        u = _u;
        v = _v;
    }
    halfplane(Line l) {
        u = l.u;
        v = l.v;
    }
    void calc_angle() {
        angle = atan2(v.y - u.y, v.x - u.x);
    }
    bool operator<(const halfplane &b) const {
        return angle < b.angle;
    }
};

const int maxp = 1005;
struct halfplanes {
    int n;
    // halfplane hp[maxp];
    vector<halfplane> hp;
    Point p[maxp];
    int que[maxp];
    int st, ed;

    void add(halfplane l) {
        hp.push_back(l);
        n++;
    }

    // 去重
    void unique() {
        int m = 1;
        for (int i = 1; i < n; i++) {
            if (sgn(hp[i].angle - hp[i - 1].angle) != 0)
                hp[m++] = hp[i];
            else if (sgn((hp[m - 1].v - hp[m - 1].u) ^ (hp[i].u - hp[m - 1].u)) > 0)
                hp[m - 1] = hp[i];
        }
        n = m;
    }

    bool halfplaneinsert() {
        for (int i = 0; i < n; i++) hp[i].calc_angle();
        sort(hp.begin(), hp.end());
        unique();
        que[st = 0] = 0;
        que[ed = 1] = 1;
        p[1] = hp[0].crosspoint(hp[1]);
        for (int i = 2; i < n; i++) {
            while (st < ed && sgn(hp[i].u.cross(hp[i].v, p[ed])) < 0) ed--;
            while (st < ed && sgn(hp[i].u.cross(hp[i].v, p[st + 1])) < 0) st++;
            que[++ed] = i;
            if (hp[i].parallel(hp[que[ed - 1]])) return false;
            p[ed] = hp[i].crosspoint(hp[que[ed - 1]]);
        }
        while (st < ed && sgn(hp[que[st]].u.cross(hp[que[st]].v, p[ed])) < 0) ed--;
        while (st < ed && sgn(hp[que[st]].u.cross(hp[que[st]].v, p[st + 1])) < 0) st++;
        if (st + 1 >= ed) return false;
        return true;
    }

    // 得到最后半平面交得到的凸多边形
    // 需要先调用halfplaneinsert() 且返回true
    void getConvex(Polygon &convex) {
        p[st] = hp[que[st]].crosspoint(hp[que[ed]]);
        convex.n = ed - st + 1;
        convex.p.resize(n);
        for (int j = st, i = 0; j <= ed; i++, j++) convex.p[i] = p[j];
    }
};
```

#### 杂项

```cpp
// 三角形面积
double triangleArea(const Point &a, const Point &b, const Point &c){
    return fabs(cross(a, b, c)) / 2.0;
}
```

## 直线凸壳

```cpp
struct Line {
    int k, b;
};
// 计算上凸壳,返回直线表达式和断点
// 断点大小为直线个数-1,brk[0]是第一个断点
pair<vector<Line>, vector<double>> upper_hull(vector<Line> lines) {
    sort(lines.begin(), lines.end(), [](const Line &a, const Line &b) {
        if (a.k != b.k)
            return a.k < b.k;
        return a.b > b.b;
    });
    vector<Line> uniq;
    for (int i = 0; i < lines.size(); ++i) {
        if (i && lines[i].k == lines[i - 1].k)
            continue;
        uniq.push_back(lines[i]);
    }
    vector<Line> stk;
    vector<double> breaks;
    for (Line& l : uniq) {
        while (stk.size() >= 2) {
            Line l1 = stk[stk.size() - 2];
            Line l2 = stk.back();
            __int128 left = (__int128)(l2.b - l.b) * (l2.k - l1.k);
            __int128 right = (__int128)(l1.b - l2.b) * (l.k - l2.k);
            if (left <= right) {
                stk.pop_back();
            } else
                break;
        }
        stk.push_back(l);
    }
    for (int i = 0; i + 1 < stk.size(); ++i) {
        int k1 = stk[i].k, b1 = stk[i].b;
        int k2 = stk[i + 1].k, b2 = stk[i + 1].b;
        double t = (double)(b1 - b2) / (k2 - k1);
        breaks.push_back(t);
    }
    return {stk, breaks};
}
// 下凸壳
pair<vector<Line>, vector<double>> lower_hull(vector<Line> lines) {
    for (Line &l : lines) {
        l.k = -l.k;
        l.b = -l.b;
    }
    auto [stk, brk] = upper_hull(lines);
    for (Line &l : stk) {
        l.k = -l.k;
        l.b = -l.b;
    }
    return {stk, brk};
}
```

## 动态凸壳

```cpp
const double eps = 1e-8;
struct Point {
    double x, y;
    Point(double x, double y) : x(x), y(y) {}
    Point operator-(const Point &a) const { return {x - a.x, y - a.y}; }
    bool operator<(const Point &a) const {
        if (x != a.x)
            return x < a.x;
        return y < a.y;
    }
};
double cross(const Point a, const Point b) {
    return a.x * b.y - b.x * a.y;
}
struct Upperhull {
    set<Point> hull;
    set<Point>::iterator pre(set<Point>::iterator it) {
        return it == hull.begin() ? hull.end() : --it;
    }
    set<Point>::iterator next(set<Point>::iterator it) {
        ++it;
        return it == hull.end() ? hull.end() : it;
    }
    bool check(Point p) {
        auto it = hull.lower_bound(p);
        if (it == hull.end())
            return false;
        if (it == hull.begin()) {
            if (abs(it->x - p.x) < eps)
                return it->y >= p.y;
            return false;
        }
        auto prev_it = pre(it);
        Point a = *prev_it - p, b = *it - p;
        return cross(a, b) <= eps;
    }
    bool erase(set<Point>::iterator it) {
        if (it == hull.end() || it == hull.begin())
            return false;
        auto prev_it = pre(it), next_it = next(it);
        if (next_it == hull.end())
            return false;
        Point a = *it - *prev_it, b = *next_it - *it;
        if (cross(a, b) >= eps) {
            hull.erase(it);
            return true;
        }
        return false;
    }
    void insert(Point p) {
        if (check(p))
            return;
        auto it = hull.insert(p).first;
        auto prev_it = pre(it), next_it = next(it);
        for (auto p = pre(it); p != hull.end() && p != hull.begin(); p = pre(it))
            if (!erase(p))
                break;
        for (auto a = next(it); a != hull.end(); a = next(it))
            if (!erase(a))
                break;
    }
};
```

## 三维几何

```cpp
const double EPS = 1e-8;
const double PI = acos(-1.0);

int sgn(double x) {
    // return (x > EPS) - (x < -EPS);
    if (fabs(x) < EPS) return 0;
    return x < 0 ? -1 : 1;
}

template <typename T>
struct Point3 {
    T x, y, z;
    constexpr Point3(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}

    constexpr Point3 operator+(const Point3& p) const { return Point3(x + p.x, y + p.y, z + p.z); }
    constexpr Point3 operator-(const Point3& p) const { return Point3(x - p.x, y - p.y, z - p.z); }
    constexpr Point3 operator*(const T& k) const { return Point3(x * k, y * k, z * k); }
    constexpr Point3 operator/(const T& k) const { return Point3(x / k, y / k, z / k); }
    Point3& operator+=(const Point3& p) { x += p.x; y += p.y; z += p.z; return *this; }
    Point3& operator-=(const Point3& p) { x -= p.x; y -= p.y; z -= p.z; return *this; }
    Point3& operator*=(const T& k) { x *= k; y *= k; z *= k; return *this; }
    Point3& operator/=(const T& k) { x /= k; y /= k; z /= k; return *this; }

    // dot product and cross product
    constexpr T operator*(const Point3& p) const { return x * p.x + y * p.y + z * p.z; }
    constexpr Point3 operator^(const Point3& p) const {
        return Point3(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
    }

    bool operator==(const Point3& p) const {
        return sgn(x - p.x) == 0 && sgn(y - p.y) == 0 && sgn(z - p.z) == 0;
    }
    bool operator!=(const Point3& p) const { return !(*this == p); }
    bool operator<(const Point3& p) const {
        if (sgn(x - p.x) != 0) return x < p.x;
        if (sgn(y - p.y) != 0) return y < p.y;
        return sgn(z - p.z) < 0;
    }

    constexpr T dis2(const Point3& p) const {
        return (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) + (z - p.z) * (z - p.z);
    }
    auto dis(const Point3& p) const {
        if constexpr (std::is_same_v<T, long double>) return sqrtl(dis2(p));
        else return sqrt(dis2(p));
    }

    T len2() const { return x * x + y * y + z * z; }
    auto len() const {
        if constexpr (std::is_same_v<T, long double>) return sqrtl(len2());
        else return sqrt(len2());
    }

    // angle (radians or degrees) ∠APB: return [0,π]
    static double angle(const Point3& a, const Point3& p, Point3& b, bool rad = true) {
        // PA*PB=|PA||PB|cos(theta)
        double costh = (a - p) * (b - p) / (p.dis(a) * p.dis(b));
        costh = std::clamp(costh, -1.0, 1.0);  // avoid precision issues
        if (rad) return acos(costh);
        return acos(costh) * 180.0 / PI;  // convert to degrees
    }

    // normalize the vector to length k
    Point3 norm(double k = 1.0) const {
        double l = len();
        if (!sgn(l)) return *this;  // origin
        k /= l;
        return Point3(x * k, y * k, z * k);
    }

    // IO
    friend std::istream& operator>>(std::istream& is, Point3& p) {
        return is >> p.x >> p.y >> p.z;
    }
    friend std::ostream& operator<<(std::ostream& os, const Point3& p) {
        return os << p.x << ' ' << p.y << ' ' << p.z;
    }
};

using Point = Point3<double>;
// using Point = Point3<long double>;
// using Point = Point3<long long>;
```

# 博弈论

## SG函数

$\textrm{SG(x)}$， $\textrm x$  是游戏的状态，若 $\textrm{SG(x)}=0$，先手必败，否则先手必胜

设后继状态为 $a_1,a_2,\dots a_p$，$\textrm{SG(x)}=\textrm{mex(SG}(a_1),\textrm{SG}(a_2)\dots \textrm{SG}(a_p))$

一个游戏的 $\textrm{SG}$ 函数值等于各个游戏 $\textrm{SG}$ 函数值的 $\textrm{nim}$ 和（异或和）

## 威佐夫博弈

两堆石子，每轮可以拿其中一堆的任意个或同时拿两堆的相同个，无法行动者败

结论：设两堆石子数量分别为为  $a,b(a\leq b)$ ，若 $a=\frac{\sqrt{5}-1}{2}(b-a)$ 则后手胜，否则先手胜。

# DP

## 数位 DP

给定区间 $[l,r]$，问区间满足条件的数有多少个，$\textrm{cal}(r)-\textrm{cal}(l-1)$

windy数（不含前导0且相邻两个数字之差至少为2），求a到b中有多少个windy数

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
//pos:当前位置 limit填数的限制 pre上一个数 lead前面是否是前导0
vector<vector<int>> dp(15,vector<int>(10,-1));
vector<int> v;
int dfs(int pos,int pre,bool lead,bool limit){
    if(pos<0) return 1;
    if(!limit&&!lead&&dp[pos][pre]!=-1){
        return dp[pos][pre];
    }
    int ans=0,up=limit?v[pos]:9;
    for(int i=0;i<=up;i++){
        if(lead){
            ans+=dfs(pos-1,i,i==0,limit&&i==up);
        }else{
            if(abs(i-pre)<2) continue;
            ans+=dfs(pos-1,i,0,limit&&i==up);
        }
    }
    return (!lead&&!limit)?dp[pos][pre]=ans:ans;
}
//计算
int cal(int x){
    v.clear();
    if(x==0) return 1;
    while(x){
        v.push_back(x%10);
        x/=10;
    }
    return dfs(v.size()-1,0,1,1);
}
void solve(){
    int a,b;
    cin>>a>>b;
    cout<<cal(b)-cal(a-1)<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

# 杂项

## 随机算法

### 模拟退火

$T$：温度

$△ T$：温度变化率，每次温度等于上一次 T*△T，一般取 0.95-0.99，模拟徐徐降温

$x$：当前选择的解

$△x$：解变动量

$x1$：当前的目标解，等于 x+△x

$△f$：当前解的函数值与目标解函数值的差值，等于 $f(x)-f(x1)$

每次的 $△x$ 在一个大小与 T 成正比的值域内随机取值。如果 $f(x1)<f(x)$，那么接受目标解 x=x 1，如果 $f(x1)>f(x)$，则以一定的概率接受，概率是 $e^{\frac{-△f}{T}}$，直到 T 趋近于 0，循环结束

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define double long double
struct edge{
    int x,y,w;
};

void solve(){
    int n;
    cin>>n;
    vector<edge> v(n);
    double xans=0,yans=0;
    for(int i=0;i<n;i++){
        cin>>v[i].x>>v[i].y>>v[i].w;
        xans+=v[i].x;
        yans+=v[i].y;
    }
    double ans=1e18;//全局能量最小值
    auto energy=[&](double x,double y){
        double res=0;
        for(int i=0;i<n;i++){
            pair<double,double> p={v[i].x-x,v[i].y-y};
            res+=sqrtl(p.first*p.first+p.second*p.second)*v[i].w;
        }
        return res;
    };//计算能量，能量越低越符合要求
    auto sa=[&](){
        double t=3000,down=0.997;//初始温度，降温系数
        double x=xans,y=yans;//当前值
        while(t>1e-14){//继续降温
            double xtmp=x+(rand()*2-RAND_MAX)*t;//新的值
            double ytmp=y+(rand()*2-RAND_MAX)*t;
            double newenergy=energy(xtmp,ytmp);//新的能量
            double delta=newenergy-ans;//新能量和全局最优能量的差值
            if(delta<0){//如果更优，接受，修改全局最优能量，全局最优答案和当前值
                xans=x=xtmp;
                yans=y=ytmp;
                ans=newenergy;
            }else if(expl(-delta/t)*RAND_MAX>rand()){//否则，以一定概率接受
                x=xtmp;//修改当前值
                y=ytmp;
            }
            t*=down;//降温
        }
    };//模拟退火
    xans/=n;
    yans/=n;
    ans=energy(xans,yans);//初始的一个答案
    while((double)clock()/CLOCKS_PER_SEC<0.8) sa();//一直多次模拟退火，直到快超时
    cout<<fixed<<setprecision(3)<<xans<<" "<<yans<<"\n";
}
signed main(){
    srand(time(0));
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 平板电视

```cpp
#include<bits/extc++.h>
using namespace __gnu_pbds;
```

## 拉链法哈希

```cpp
gp_hash_table <int,int> f;
```

## 红黑树

```cpp
__gnu_pbds::tree<Key, Mapped, Cmp_Fn = std::less<Key>, Tag = rb_tree_tag,
                  Node_Update = null_tree_node_update,
                  Allocator = std::allocator<char> >
//如果要用order_of_key或find_by_order，node_update需要使用tree_order_statistics_node_update
```

```cpp
__gnu_pbds::tree<std::pair<int, int>, __gnu_pbds::null_type,
                 std::less<std::pair<int, int> >, __gnu_pbds::rb_tree_tag,
                 __gnu_pbds::tree_order_statistics_node_update>
    trr;
```

```cpp
// insert(x)：向树中插入一个元素 x，返回 std::pair<point_iterator, bool>。
// erase(x)：从树中删除一个元素/迭代器 x，返回一个 bool 表明是否删除成功。
// order_of_key(x)：返回 x 以 Cmp_Fn 比较的排名，**0 开始**。
// find_by_order(x)：返回 Cmp_Fn 比较的排名所对应元素的迭代器。
// lower_bound(x)：以 Cmp_Fn 比较做 lower_bound，返回迭代器。
// upper_bound(x)：以 Cmp_Fn 比较做 upper_bound，返回迭代器。
// join(x)：将 x 树并入当前树，前提是两棵树的类型一样，x 树被删除。
// split(x, b)：以 Cmp_Fn 比较，小于等于 x 的属于当前树，其余的属于 b 树。
// empty()：返回是否为空。
// size()：返回大小。

// 元素不可重，可以使用 pair<int,int>加入时间戳来使元素可重

//插入
tr.insert({x,++cnt});
//删除
auto it=tr.lower_bound({x,0});
if(it!=tr.end()) tr.erase(it);
//查找元素排名
tr.order_of_key({x,0})+1;
//根据排名查找元素
auto it=tr.find_by_order(x-1);
if(it==tr.end()) continue;
it->first;
//前驱
auto it=tr.lower_bound({x,0});
if(it==tr.begin()) continue;
it--;
it->first;
//后继
auto it=tr.upper_bound({x,INF});
if(it==tr.end()) continue;
it->first;
```

## 笛卡尔树

son first左儿子second右儿子
p符合堆的性质
下标符合二叉搜索树的性质

```cpp
int n;
cin>>n;
vector<int> p(n+1);
vector<pair<int,int>> son(n+1);
for(int i=1;i<=n;i++) cin>>p[i];
stack<int> st;
int rt=-1;
for(int i=1;i<=n;i++){
    if(st.empty()) rt=i;
    else{
        int pre=-1;
        while(!st.empty()&&p[st.top()]>p[i]){
            pre=st.top();
            st.pop();
        }
        if(st.empty()){
            rt=i;
            son[rt].first=pre;
        }else{
            son[i].first=son[st.top()].second;
            son[st.top()].second=i;
        }
    }
    st.push(i);
}
```

## wqs二分

一般用于求恰好选$k$个物品的答案

设$g[i]$表示恰好选$i$个物品的答案，需要满足$(i,dp[i])$是个凸包。

假设$g[i]$是个下凸函数：

用直线$y=kx+b$去切这个凸包，切于点$x_0$，$x_0$越大，$k$越大，b越小。可以得出此时的截距$b=g(x_0)-kx_0$。算出来切于答案点的$k$和b。

给你一个无向带权连通图，每条边是黑色或白色。让你求一棵最小权的恰好有$need$条白色边的生成树。

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int INF=1e18;
struct DSU{
    vector<int> fa;
    vector<int> rank;
    int n;
    DSU(int n){
        this->n=n;
        fa=vector<int>(n+1);
        rank=vector<int>(n+1,0);
        for(int i=0;i<=n;i++) fa[i]=i;
    }
    int find(int x){
        return fa[x]==x?x:fa[x]=find(fa[x]);
    }
    bool merge(int x,int y){
        int fax=find(x);
        int fay=find(y);
        if(fax==fay) return 0;
        if(rank[fax]<rank[fay]) fa[fax]=fay;
        else{
            fa[fay]=fax;
            if(rank[fax]==rank[fay]) rank[fax]++;
        }
        return 1;
    }
};
void solve(){
    int n,m,k;
    cin>>n>>m>>k;
    vector<array<int,4>> v(m);
    for(int i=0;i<m;i++){
        cin>>v[i][0]>>v[i][1]>>v[i][2]>>v[i][3];
    }
    int l=-105,r=105,ans1,ans2;
    auto check=[&](int k)->pair<int,int>{
        DSU dsu(n);
        for(int i=0;i<m;i++){
            if(v[i][3]==0){
                v[i][2]-=k;
            }
        }
        sort(v.begin(),v.end(),[&](const auto &a,const auto &b){
            if(a[2]!=b[2]) return a[2]<b[2];
            return a[3]<b[3];
        });
        pair<int,int> ans={INF,INF};
        int sum=0,cnt=0;
        for(int i=0;i<m;i++){
            if(dsu.merge(v[i][0],v[i][1])){
                sum+=v[i][2];
                if(v[i][3]==0) cnt++;
            }
        }
        ans=min(ans,{sum,-cnt});
        for(int i=0;i<m;i++){
            if(v[i][3]==0){
                v[i][2]+=k;
            }
        }
        return ans;
    };
    while(l<=r){
        int mid=(l+r)/2;
        auto [a,b]=check(mid);
        if(-b>=k){
            ans1=a;
            ans2=mid;
            r=mid-1;
        }else l=mid+1;
    }
    cout<<ans1+ans2*k<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## 快读快写

```cpp
void get(int &x){
    x=0;
    char ch=getchar();
    while(ch<'0'||ch>'9'){
        ch=getchar();
    }
    while(ch>='0'&&ch<='9'){
        x=x*10+ch-'0';
        ch=getchar();
    }
}
void print(__int128 x){
    if(x==0){
        putchar('0');
        return;
    }
    stack<char> st;
    while(x){
        st.push(x%10+'0');
        x/=10;
    }
    while(!st.empty()){
        putchar(st.top());
        st.pop();
    }
}
```

## 火车头

```cpp
#define fastcall __attribute__((optimize("-O3")))
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
```

## 日期公式

```cpp
// 从公元 1 年 1 月 1 日经过了多少天
int getday(int y, int m, int d) {
    if (m < 3)
        --y, m += 12;
    return (365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + d - 307);
}
// 公元 1 年 1 月 1 日往后数 n 天是哪一天
void date(int n, int &y, int &m, int &d) {
    n += 429 + ((4 * n + 1227) / 146097 + 1) * 3 / 4;
    y = (4 * n - 489) / 1461, n -= y * 1461 / 4;
    m = (5 * n - 1) / 153, d = n - m * 153 / 5;
    if (--m > 12)
        m -= 12, ++y;  
}
```

## 枚举子集

复杂度 $\mathcal{O}(3^n)$

```cpp
for (int s = u; s; s = (s - 1) & u) {
    // s 是 u 的一个非空子集 
}
```

## cdq分治

第一维排序，每次cdq分治统计按照第一维排序时，下标$[l,mid]$对$[mid+1,r]$的贡献。

一般第二维归并（保证了第二维小的在前面），第三维bit or 线段树（查询比当前第三维小的）。

cdq分治的时候，算完贡献，记得清空BIT。

选择完一个元素，记得i++/j++。

cdq分治完记得把临时数组的排序结果移回原数组。

cdq分治优化dp这类的题目，可能$[mid+1,r]$这个区间，依赖于$[l,mid]$这个区间的结果，此时的执行顺序应为

1. 递归$[l,mid]$
2. 处理$[l,mid]$对$[mid+1,r]$的贡献
3. 再递归$[mid+1,r]$的贡献

第二步的时候，先对右半部分第二维排好序，再归并，然后把右半部分还原回按照第一维排序的形式，再进行第三步操作。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
struct BIT{
    vector<int> tree;
    int n;
    inline int lowbit(int x){
        return x&(-x);
    }
    BIT(int n){
        this->n=n;
        tree.resize(n+1,0);
    }
    void update(int x,int delta){
        ++x;
        for(int i=x;i<=n;i+=lowbit(i)) tree[i]+=delta; 
    }
    int query(int x,int y){
        ++x,++y;
        if(x>y) return 0;
        int ans=0;
        for(int i=y;i>=1;i-=lowbit(i)) ans+=tree[i];
        for(int i=x-1;i>=1;i-=lowbit(i)) ans-=tree[i];
        return ans;
    }
};
void solve(){
    vector<array<int,5>> v;
    map<array<int,3>,int> mp;
    int op,cnt=0,w;
    for(int i=1;;i++){
        cin>>op;
        if(op==3) break;
        if(op==0){
            cin>>w;
            continue;
        }
        if(op==1){
            int x,y,a;
            cin>>x>>y>>a;
            v.push_back({0,i,x,y,a});
        }else{
            int x1,y1,x2,y2;
            cin>>x1>>y1>>x2>>y2;
            v.push_back({1,i,x2,y2,cnt});
            v.push_back({1,i,x1-1,y1-1,cnt});
            v.push_back({-1,i,x1-1,y2,cnt});
            v.push_back({-1,i,x2,y1-1,cnt});
            cnt++;
        }
    }
    BIT bit(2e5);
    vector<int> ans(cnt);
    sort(v.begin(),v.end(),[](const auto &a,const auto &b){
        if(a[2]!=b[2]) return a[2]<b[2];
        else if(a[3]!=b[3]) return a[3]<b[3];
        else return a[1]<b[1];
    });
    vector<array<int,5>> tmp(v.size());
    auto cdq=[&](auto &&self,int l,int r){
        if(l>=r) return;
        int mid=l+(r-l>>1);
        self(self,l,mid);
        self(self,mid+1,r);
        queue<array<int,2>> q;
        for(int i=l,j=mid+1,cnt=l;cnt<=r;cnt++){
            if(i<=mid&&j<=r){
                if(v[i][3]<v[j][3]||v[i][3]==v[j][3]&&v[i][1]<v[j][1]){
                    if(v[i][0]==0){
                        bit.update(v[i][1],v[i][4]);
                        q.push({v[i][1],v[i][4]});
                    }
                    tmp[cnt]=move(v[i++]);
                }else{
                    if(v[j][0]){
                        ans[v[j][4]]+=v[j][0]*bit.query(1,v[j][1]-1);
                    }
                    tmp[cnt]=move(v[j++]);
                }
            }else if(i<=mid){
                if(v[i][0]==0){
                    bit.update(v[i][1],v[i][4]);
                    q.push({v[i][1],v[i][4]});
                }
                tmp[cnt]=move(v[i++]);
            }else{
                if(v[j][0]){
                    ans[v[j][4]]+=v[j][0]*bit.query(1,v[j][1]-1);
                }
                tmp[cnt]=move(v[j++]);
            }
        }
        for(int i=l;i<=r;i++){
            v[i]=move(tmp[i]);
        }
        while(!q.empty()){
            auto [a,b]=q.front();
            q.pop();
            bit.update(a,-b);
        }
    };
    cdq(cdq,0,v.size()-1);
    for(int i=0;i<ans.size();i++) cout<<ans[i]<<"\n";
}
signed main(){
    cin.tie(nullptr)->sync_with_stdio(0);
    int t=1;
    //cin>>t;
    while(t--) solve();
    return 0;
}
```

## CPH

Shell

```sh
#!/bin/bash
g++-14 -O2 main.cpp -o a.out
if [ $? -ne 0 ]; then exit; fi
s=$(python3 -c "import time; print(int(time.time()*1000))")
./a.out < in.txt > out.txt 2> err.txt
res=$?
e=$(python3 -c "import time; print(int(time.time()*1000))")
cat out.txt
echo -e "\n--- err ---\n"
cat err.txt
echo -e "\nTime: $((e-s))ms"
if [ $res -ne 0 ]; then
    echo "RE!!!!!!!"
fi
rm a.out out.txt err.txt
read -n 1 -s
```

PowerShell

```powershell
clear
g++ -O2 main.cpp -o a.exe
if (!$?) { exit }
$t = Measure-Command { cmd /c "a.exe < in.txt > out 2> err" }
gc out; echo "`n--- err ---`n"; gc err; echo "`nTime: $([int]$t.TotalMilliseconds)ms"
if ($LASTEXITCODE -ne 0) {echo "RE!!!!!!!"}
rm a.exe, out, err -ErrorAction SilentlyContinue
# 若不需要按下回车后退出可以不打最后一行
$null = [Console]::ReadKey()
```

## 对拍

Shell

```sh
#!/bin/bash
clear
mkdir -p data
trap "rm -f m a r" EXIT
g++ maker.cpp -o m; g++ main.cpp -o a; g++ right.cpp -o r
cnt=1
while true; do
  ./m > data/in; ./a < data/in > data/out; ./r < data/in > data/ans
  if diff data/out data/ans > /dev/null; then
    echo "AC #$cnt"; ((cnt++))
  else
    echo "WA!"
    echo "--- Data ---"; cat data/in
    echo "--- Main ---"; cat data/out
    echo "--- Right ---"; cat data/ans
    break
  fi
done
```

PowerShell

```powershell
clear
g++ maker.cpp -o m.exe; g++ main.cpp -o a.exe; g++ right.cpp -o r.exe
if (! (Test-Path data)) { mkdir data }
try {
    $n = 1
    while ($true) {
        ./m.exe > data/in
        gc data/in | ./a.exe > data/out
        gc data/in | ./r.exe > data/ans
        if ((gc data/out -Raw) -eq (gc data/ans -Raw)) {
            echo "AC #$n"; $n++
        } else {
            echo "WA!"
            echo "--- Data ---"; gc data/in
            echo "--- Main ---"; gc data/out
            echo "--- Right ---"; gc data/ans
            break
        }
    }
} finally {
    rm m.exe, a.exe, r.exe -ErrorAction SilentlyContinue
}
```

## 随机数哈希

```cpp
struct Hash {
    static inline const uint64_t R = chrono::steady_clock::now().time_since_epoch().count();
    static uint64_t mix(uint64_t x) {
        x += R;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
    }
    size_t operator()(long long x) const {
        return mix(x);
    }
    size_t operator()(const pair<long long, long long> &p) const {
        return mix((uint64_t)p.first ^ ((uint64_t)p.second << 1));
    }
};
```

## 快速模

```cpp
struct Barrett {
    int d;
    __uint128_t m;
    void init(int _d) {
        d = _d, m = (((__uint128_t)(1) << 64) / d);
    }
} mod;
int operator%(int a, Barrett mod) {
    int b = abs(a);
    int w = (mod.m * b) >> 64;
    w = b - w * mod.d;
    if (w >= mod.d)
        w -= mod.d;
    if (a < 0) {
        if (w == 0)
            return 0;
        w = mod.d - w;
    }
    return w;
}
```

## ModInt

```cpp
template<int MOD>
struct ModInt{
    int v;
    ModInt(int x=0){
        v=(x%MOD+MOD)%MOD;
    }
    ModInt& operator+=(const ModInt &o){
        v+=o.v;
        if(v>=MOD) v-=MOD;
        return *this;
    }
    ModInt& operator-=(const ModInt &o){
        v-=o.v;
        if(v<0) v+=MOD;
        return *this;
    }
    ModInt& operator*=(const ModInt &o){
        v=v*o.v%MOD;
        return *this;
    }
    ModInt qpow(int k) const{
        ModInt res=1,base=*this;
        while(k){
            if(k&1) res*=base;
            base*=base;
            k>>=1;
        }
        return res;
    }
    ModInt inv() const{
        return qpow(MOD-2);
    }
    ModInt& operator/=(const ModInt &o){
        return *this*=o.inv();
    }
    bool operator==(ModInt a){
        return a.v==v;
    }
    bool operator!=(ModInt a){
        return !(a.v==v);
    }
    friend ModInt operator+(ModInt a,const ModInt &b){
        return a+=b;
    }
    friend ModInt operator-(ModInt a,const ModInt &b){
        return a-=b;
    }
    friend ModInt operator*(ModInt a,const ModInt &b){
        return a*=b;
    }
    friend ModInt operator/(ModInt a,const ModInt &b){
        return a/=b;
    }
    friend ostream& operator<<(ostream &os,const ModInt &x){
        return os<<x.v;
    }
    friend istream& operator>>(istream &is,ModInt &x){
        int v;
        is>>v;
        x=ModInt(v);
        return is;
    }
};
using mint=ModInt<998244353>;
```