## 数据结构

### 树状数组

区间和

```c++
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

区间最值

```cpp
struct BIT{
    vector<int> tree;
    vector<int> raw;
    int n;
    inline int lowbit(int x){
        return x&(-x);
    }
    BIT(int n){
        this->n=n;
        tree.resize(n+1,0);
        raw.resize(n+1,0);
    }
    void update(int x,int y){
        raw[x]=y;
        for(int i=x;i<=n;i+=lowbit(i)){
            tree[i]=raw[i];
            for(int j=1;j<lowbit(i);j<<=1){
                tree[i]=max(tree[i],tree[i-j]);
            }
        }
    }
    int query(int x,int y){
        if(x>y) return 0;
        int ans=0;
        while(x<=y){
            int nx=y-lowbit(y)+1;
            if(nx>=x){
                ans=max(ans,tree[y]);
                y=nx-1;
            }else{
                ans=max(ans,raw[y]);
                --y;
            }
        }
        return ans;
    }
};
```



### 线段树

```c++
struct SegmentTree{
    struct edge{
        int sum;
        edge(){
            sum=0;
        }
    };
    vector<int> lazy;
    vector<edge> node;
    int n;
    void pushup(int id,int l,int r){
        node[id].sum=node[id<<1].sum+node[id<<1|1].sum;
    }
    void pushdown(int id,int l,int r){
        if(lazy[id]){
            int mid=l+(r-l>>1);
            lazy[id<<1]+=lazy[id];
            lazy[id<<1|1]+=lazy[id];
            node[id<<1].sum+=(mid-l+1)*lazy[id];
            node[id<<1|1].sum+=(r-mid)*lazy[id];
            lazy[id]=0;
        }
    }
    SegmentTree(int n):n(n){
        node.resize((n<<2)+5);
        lazy.assign((n<<2+5),0);
    }
    SegmentTree(){}
    void init(vector<int> &v){
        function<void(int,int,int)> buildtree=[&](int id,int l,int r){
            lazy[id]=0;
            if(l==r){
                node[id].sum=v[l];
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
        lazy.assign((n<<2+5),0);
        init(v);
    }
    void update(int id,int l,int r,int x,int y,int delta){
        if(x<=l&&r<=y){
            lazy[id]+=delta;
            node[id].sum+=delta*(r-l+1);
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
        if(y>mid) ans+=query(id<<1|1,mid+1,r,x,y);
        return ans;
    }
};
```

### 线段树标记永久化

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



### 并查集

```c++
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
        fa[x]=y;
        sz[x]=1;
        return 1;
    }
};
```

```c++
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
        if(x==y) return 0;
        if(rank[fax]<rank[fay]) fa[fax]=fay;
        else{
            fa[fay]=fax;
            if(rank[fax]==rank[fay]) rank[fax]++;
        }
        return 1;
    }
};
```

### 可持久化线段树

开40倍空间

```cpp
struct PresidentTree{
    vector<int> node;
    vector<int> lson,rson;
    vector<int> head;
    int n;
    int cnt;
    PresidentTree(int n,vector<int> &v):n(n){
        node.resize(40*n);
        lson.resize(40*n);
        rson.resize(40*n);
        cnt=0;
        function<int(int,int)> buildtree=[&](int l,int r){
            int now=++cnt;
            if(l==r){
                node[now]=v[l];
                return now;
            }
            int mid=l+(r-l>>1);
            lson[now]=buildtree(l,mid);
            rson[now]=buildtree(mid+1,r);
            return now;
        };
        head.push_back(buildtree(1,n));
    }
    void update(int nowid,int baseid,int x,int y){
        function<int(int,int,int,int,int)> updatenode=[&](int base,int l,int r,int x,int y){
            int now=++cnt;
            if(l==r){
                node[now]=y;
                return now;
            }
            int mid=l+(r-l>>1);
            if(x<=mid){
                lson[now]=updatenode(lson[base],l,mid,x,y);
                rson[now]=rson[base];
            }else{
                lson[now]=lson[base];
                rson[now]=updatenode(rson[base],mid+1,r,x,y);
            }
            return now;
        };
        head.push_back(updatenode(head[baseid],1,n,x,y));
    }
    int query(int id,int x){
        function<int(int,int,int,int)> querynode=[&](int base,int l,int r,int x){
            if(l==r) return node[base];
            int mid=l+(r-l>>1);
            if(x<=mid) return querynode(lson[base],l,mid,x);
            else return querynode(rson[base],mid+1,r,x);
        };
        return querynode(head[id],1,n,x);
    }
};
```

### 可持久化01trietree

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





### ST 表

```cpp
struct ST{
    static vector<int> Log2;
    vector<vector<int>> dp;
    ST(int n,vector<int> &v){
        for(int i=Log2.size();i<=n;i++){
            if(i==0) Log2.push_back(0);
            else if(i==1) Log2.push_back(0);
            else Log2.push_back(Log2[i>>1]+1);
        }
        dp.resize(n+1);
        for(int i=1;i<=n;i++){
            dp[i].resize(20);
            dp[i][0]=v[i];
        }
        for(int i=1;i<=18;i++){
            for(int j=1;j+(1ll<<i)-1<=n;j++){
                dp[j][i]=max(dp[j][i-1],dp[j+(1ll<<i-1)][i-1]);
            }
        }
    }
    int query(int l,int r){
        int k=Log2[r-l+1];
        return max(dp[l][k],dp[r-(1ll<<k)+1][k]);
    }
};
vector<int> ST::Log2;
```

### 权值线段树

```cpp
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
```

### 树套树

#### 树状数组套权值线段树

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

#### 线段树套平衡树

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



### 猫树

$O(nlogn)$次合并，查询时$O(1)$次合并查询

## 字符串

### 序列自动机

$nxt[i][j]$表示从第i个位置开始，字符串j出现的第一个位置

```c++
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

### AC自动机

多模式串匹配
```c++
template<int Base>
struct ACAutomaton{
    vector<array<int,26>> tree;
    vector<int> ed;
    vector<int> fail;
    vector<vector<int>> id;
    vector<int> exist;
    vector<int> tag;
    vector<int> in;
    int cnt;
    void insert(string &s,int num){
        int u=0;
        for(char &c:s){
            if(!tree[u][c-Base]){
                tree[u][c-Base]=++cnt;
                tree.emplace_back();
                tree.back().fill(0);
                ed.emplace_back(0);
                fail.emplace_back(0);
                id.emplace_back();
                in.emplace_back(0);
            }
            u=tree[u][c-Base];
        }
        ed[u]++;
        id[u].push_back(num);
    }
    void build(){
        queue<int> q;
        for(int i=0;i<26;i++){
            if(tree[0][i]) q.push(tree[0][i]); 
        }
        while(!q.empty()){
            int u=q.front();
            q.pop();
            for(int i=0;i<26;i++){
                if(tree[u][i]){
                    fail[tree[u][i]]=tree[fail[u]][i];
                    in[tree[fail[u]][i]]++;
                    q.push(tree[u][i]);
                }else{
                    tree[u][i]=tree[fail[u]][i];
                }
            }
        }
    }
    void topo(){
        queue<int> q;
        for(int i=1;i<=cnt;i++){
            if(!in[i]) q.push(i);
        }
        while(!q.empty()){
            int f=q.front();
            q.pop();
            for(int &p:id[f]) exist[p]=tag[f];
            int u=fail[f];
            tag[u]+=tag[f];
            if(!(--in[u])) q.push(u);
        }
    }
    void query(string &s){
        tag.resize(cnt+1,0);
        int u=0,ans=0;
        for(int i=0;i<s.size();i++){
            u=tree[u][s[i]-Base];
            tag[u]++;
        }
        topo();
    }
    ACAutomaton(vector<string> &v){
        tree.resize(1);
        in.resize(1,0);
        exist.resize(v.size(),0);
        id.resize(v.size());
        cnt=0;
        tree.back().fill(0);
        fail.push_back(0);
        ed.push_back(0);
        for(int i=0;i<v.size();i++) insert(v[i],i);
        build();
    }
};
```

### Manacher

求回文串长度

(s的下标+1)*2对应p数组

p数组的值-1对应了回文串长度

```c++
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



### Trie Tree

```c++
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

### 扩展KMP

对于一个长度为n的字符串，定义函数z[i]，表示s和s[i,n-1] (即以s[i]开头的后缀)的最长公共前缀（LCP）的长度，则z被称为s的z函数，其中z[0]=0。

```c++
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

### 前缀函数

π[i]表示子串[0,i]最长的相等的真前缀与真后缀的长度

其中π[0]=0

```c++
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

### KMP函数

给定一个文本text和一个字符串pattern，找到并展示s在t中的所有出现位置，时间复杂度O(n+m)

```c++
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



### 字符串哈希

```c++
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

## 数学

### 高精度

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



### 矩阵相关

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
        vector<vector<double>> ans(n,vector<double>(e.m,0));
        for(int i=0;i<n;i++){
            for(int j=0;j<e.m;j++){
                for(int k=0;k<m;k++){
                    ans[i][j]+=v[i][k]*e.v[k][j];
                }
            }
        }
        return ans;
    };
    //高斯消元
    //无解-1，无穷解0，有唯一解1
    int Gauss(){
        int column=0;
        for(int i=0;i<n;i++){
            while(column<m){
                int line=i;
                double maxn=v[i][column];
                for(int j=i+1;j<n;j++){
                    if(fabs(v[j][column])>maxn){
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
        return ans;
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
                    if(fabs(v[j][column])>maxn){
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
                    for(int z=column;z<m;z++){
                        v[j][z]^=k&v[i][z];
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



### 卡特兰数

有一个大小为n*n的方格图，左下角为(0,0)，右上角为(n,n)，从左下角开始每次只能向右或者向上走一个单位，不能走到y=x上方（但可以触碰），有几种可能的路径

递推式：
$$
H_n=\frac{\binom{2n}{n}}{n+1}(n\ge2,n\in N_+)
$$

$$
H_n=\begin{cases}
\sum_1^n H_{i-1}H_{n-i},n\ge2,n\in N_+\\
1,n=0,1
\end{cases}
$$

$$
H_n=\frac{H_{n-1}(4n-2)}{n+1}
$$

$$
H_n=\binom{2n}{n}-\binom{2n}{n-1}
$$



### MillerRabin

判断某个数是否是质数

```c++
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

### PollardRho

判断质数（使用millerrabin判断），计算因子

```c++
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

### 超快质因数分解&求约数

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



### 线性筛质数

```c++
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

### 线性筛欧拉函数

```c++
struct EulerSieve{
    vector<int> prime;
    vector<bool> isPrime;
    vector<int> phi;
    int n;
    EulerSieve(int n){
        this->n=n;
        isPrime=vector<bool>(n+1,1);
        phi=vector<int>(n+1);
        isPrime[1]=0;
        for(int i=2;i<=n;i++){
            if(isPrime[i]){
                prime.push_back(i);
                phi[i]=i-1;
            }
            for(int &p:prime){
                if(i*p>n) break;
                isPrime[i*p]=0;
                if(i%p==0){
                    phi[i*p]=phi[i]*p;
                    break;
                }else{
                    phi[i*p]=phi[i]*phi[p];
                }
            }
        }
    }
};
```

### 直接求欧拉函数

```c++
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
### 组合数学

```c++
template<int MOD>
struct Comb{
    vector<int> jc,ijc;
    int quickpow(int x,int y){
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
        if(n<=0||k-1<0||n<k) return 0;
        return jc[n]*ijc[k-1]%MOD;
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

### Lucas

用于求解问题规模很大，而模数是一个不大的质数的时候的组合数问题，p为质数
$$
C^m_n~mod~p=
\begin{cases}
1~~m=0\\
C_{[\frac{n}{p}]}^{[\frac{m}{p}]}*C^{m~mod~p}_{n~mod~p}~mod~p
\end{cases}
$$

### 第二类斯特林数

将n个两两不同的元素，划分为k个互不区分的非空子集的方案数

### 康托展开

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
        void pushup(int id,int l,int r){
            tree[id]=tree[id<<1]+tree[id<<1|1];
        }
        void realupdate(int id,int l,int r,int x,int delta){
            if(l==r){
                tree[id]+=delta;
                return;
            }
            int mid=l+(r-l>>1);
            if(x<=mid) realupdate(id<<1,l,mid,x,delta);
            else realupdate(id<<1|1,mid+1,r,x,delta);
            pushup(id,l,r);
        }
        int realquerysum(int id,int l,int r,int x,int y){
            if(x<=l&&r<=y) return tree[id];
            int mid=l+(r-l>>1);
            int ans=0;
            if(x<=mid) ans+=realquerysum(id<<1,l,mid,x,y);
            if(y>mid) ans+=realquerysum(id<<1|1,mid+1,r,x,y);
            return ans;
        }
        int realquerykth(int id,int l,int r,int k){
            if(l==r) return l;
            int mid=l+(r-l>>1);
            int lsum=mid-l+1-tree[id<<1];
            if(k<=lsum) return realquerykth(id<<1,l,mid,k);
            else return realquerykth(id<<1|1,mid+1,r,k-lsum);
        }
        void update(int x,int delta){
            realupdate(1,1,n,x,delta);
        }
        int querysum(int x,int y){
            return realquerysum(1,1,n,x,y);
        }
        int querykth(int k){
            return realquerykth(1,1,n,k);
        }
    };
    vector<int> fac;
    int n;
    const int mod;
    SegmentTree tree;
    Cantor(int n,int mod=1e20):n(n),mod(mod),tree(n){
        fac.resize(n+1);
        fac[0]=1;
        for(int i=1;i<=n;i++) fac[i]=fac[i-1]*i%mod;
    }
    //排名
    int get(vector<int> &v){
        int ans=0;
        int sz=v.size();
        for(int i=0;i<sz;i++){
            ans=(ans+fac[sz-i-1]*((v[i]-1-tree.querysum(1,v[i]-1))+mod)%mod)%mod;
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
### 快速幂

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

### 线性基

解决异或问题。原序列里的每一个数都可以由线性基里面的一些数异或得到，线性基里面任意一些数异或起来不等于0，线性基里面的数的个数唯一，且数的个数是最小的。

##### 插入

d[i]存最高位1在第i位的数

对于每个数x，假设最高位的1在第i位，如果d[i]等于0，d[i]=x，插入完成，否则x^=d[i]，继续插入

```cpp
void insert(int x){
    for(int i=51;i>=0;i--){
        if(x>>i&1){
            if(p[i]) x^=p[i];
            else{
                p[i]=x;
                break;
            }
        }
    }
}
```

##### 查询某个数能否被异或出来

```cpp
bool ask(int x){
    for(int i=51;i>=0;i--){
        if(x>>i&1){
            x^=p[i];
        }
    }
    return x==0;
}
```

##### 查询异或最大值

```cpp
int askmx(int x){
    int ans=0;
    for(int i=51;i>=0;i--){
        if((ans^p[i])>ans) ans^=p[i];    
    }
    return ans;
}
```

##### 查询异或最小值

```cpp
int askminn(int x){
    for(int i=51;i>=0;i--){
        if(!p[i]){
            return p[i];
        } 
    }
    for(int i=0;i<=51;i++){
        if(p[i]) return p[i];
    }
}
```

##### 查询异或第k小

重构一个各个位之间互不影响的d数组

```cpp
void rebuild(){
    for(int i=51;i>=0;i--){
        for(int j=i-1;j>=0;j--){
            if(p[i]>>j&1) p[i]^=p[j];
        }
    }
    for(int i=0;i<=51;i++){
        if(p[i]){
            d.push_back(p[i]);
        }
    }
}
```

```cpp
int querykth(int k){
    if(count(p.begin(),p.end(),0)){
        if(k==1) return 0;
        k--;
    }
    int ans=0;
    for(int i=d.size()-1;i>=0;i--){
        if(k>>i&1) ans^=d[i];
    }
    return ans;
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
        while(l<query[i].x){
            upd(l++);
        }
        while(l>query[i].x){
            upd(--l);
        }
        while(r<query[i].y){
            upd(++r);
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

用于棋盘状态压缩，每个位置的每个棋子状态（例如（1,1）位置为黑棋）使用mt19937_64赋予一个随机值，最后整个棋盘的状态等于所有棋子的异或和

## 图论

### Cayley 凯莱定理

一个完全图不同的生成树的数量有$n^{n-2}$种

### 最小生成树

#### Kruskal

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

### 判负环

SPFA判负环，一旦一个点入队次数大于等于n，即存在负环

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
                    if(++in[to]>n) return 1;
                    if(!vis[to]){
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

### 网络流

网络指一个特殊的有向图G=(V,E)，其与一般有向图的不同之处在于有容量和源汇点（源点s，汇点t）。任意节点净流量为0，且流经该边的流量不得超过该边的容量（边(u,v)的容量记作c(u,v)）。定义f的流量为源点s的净流量。

#### 最大流

使流量f尽可能大，dinic算法，时间复杂度$O(mn^2)$

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

#### 最小割

网络G=(V,E)的一个割{S,T}，S和T是点的一个划分，$s\in S,t\in T$，{S,T}的容量=${\textstyle \sum_{u\in S} \sum_{v\in T}c(u,v)}$

最小割要找到一个割，使得容量尽可能小

根据最大流最小割定理，最大流=最小割，直接套用最大流即可

#### 最小费用最大流

在网络上对每条边(u,v)给定一个权值w(u,v)，称为费用，含义是单位流量通过(u,v)所花费的代价，对于G所有可能的最大流中总费用最小的为最小费用最大流，SSP算法，O(nmf)，其中f为网络最大流

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

### 差分约束

n元一次不等式组，包含n个变量x1……xn，以及m个约束条件，形如xi-xj<=ck，其中ck为常量。令dis0等于0，0向所有的点连一条点权为0的边，dis[i]<=dis[j]+ck，则j到i连一条长度为ck的边。如果存在负环则无解。

### 强连通分量

```cpp
function<void(int)> tarjan=[&](int x){
        dfn[x]=low[x]=++cnt;
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
            ans.push_back({});
            while(!st.empty()&&st.top()!=x){
                int f=st.top();
                st.pop();
                ans.back().push_back(f);
                belong[f]=ans.size();
                instack[f]=0;
            }
            st.pop();
            ans.back().push_back(x);
            instack[x]=0;
            belong[x]=ans.size();
        }
    };
    for(int i=1;i<=n;i++){
        if(!dfn[i]) tarjan(i);
    }
```

### 割点与桥

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



### 双联通分量

#### 边双联通分量

先求出桥，把割点删去，剩下的极大联通子图就是边双联通分量，不能用常规方法存图

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

#### 点双联通分量

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



### 最短路

#### SPFA

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



#### dijkstra

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



#### Johnson

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



### 普通环计数

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

### 三元环计数

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

### 2-SAT问题

n个集合，每个集合有两个元素，已知若干个<a,b>，表示a与b矛盾（a，b不属于同一个集合），需要从每个集合中选择一个元素，判断能否选n个两两不矛盾元素。

a1和b2有矛盾，则建有向边a1->b1,b2->a2，tarjan缩点判断是否有一个集合中的两个元素都在同一个强联通块，如果是则不可能。

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

### 树链剖分

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
        lazy.assign((n<<2+5),0);
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
        lazy.assign((n<<2+5),0);
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

### 表达式树

中序表达式转换为逆波兰式

符号优先级：^:3，*/:2，+-:1，():0

s[i]中：

①如果是数字，压入结果栈

②如果是乘方，压入符号栈

③如果是+-*/，将符号栈栈顶比他优先级高或相同的符号一一弹出，并压入结果栈，然后将s[i]压入符号栈

④如果是左括号，压入符号栈

⑤如果是右括号，将符号栈顶部第一个左括号之前的符号弹出并压入结果栈，并弹出左括号

最后将符号栈中剩余的符号都弹出并压入结果栈

### 树哈希

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



## 计算几何

```cpp
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
    //极角排序
    inline static bool sortPointAngle(const Point &a,const Point &b){
        if(a.quad()!=b.quad()) return a.quad()<b.quad();
        return (a^b)>0;
    }
    //模长
    inline double norm() const{
        return sqrtl(x*x+y*y);
    }
    //向量方向
    //1 a在b逆时针方向
    //0 同向或反向
    //2 a在b顺时针方向
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
    inline double dis(const Point &e){
        Point c=(*this)-e;
        return c.norm();
    }
    double getangle(const Point &e) const{
        return fabs(atan2l(*this^e,*this*e));
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
};
//要先getConvex求凸包，其他的才能用
struct Polygon{
    vector<Point> p;
    vector<Point> convexhull;
    int n;
    Polygon(int n,vector<Point> &v):n(n),p(v){} 
    Polygon(int n):n(n),p(n){}
    void input(){
        for(int i=0;i<n;i++){
            cin>>p[i].x>>p[i].y;
        }
    }
    void getConvex(){
        sort(p.begin(),p.end(),Point::sortxupyup);
        p.erase(unique(p.begin(),p.end(),[](const Point &a,const Point &b){
            return a.x==b.x&&a.y==b.y;
        }),p.end());
        n=p.size();
        if(n==0) return;
        if(n==1){
            convexhull.push_back(p.front());
            convexhull.push_back(p.front());
            return;
        }
        vector<int> st(2*n+5,0);
        vector<bool> used(n,0);
        int tp=0;
        st[++tp]=0;
        for(int i=1;i<n;i++){
            while(tp>=2&&((p[st[tp]]-p[st[tp-1]])^(p[i]-p[st[tp]]))<=0){
                used[st[tp--]]=0;
            }
            used[i]=1;
            st[++tp]=i;
        }
        int tmp=tp;//下凸壳大小
        for(int i=n-2;i>=0;i--){
            if(!used[i]){
                while(tp>tmp&&((p[st[tp]]-p[st[tp-1]])^(p[i]-p[st[tp]]))<=0){
                    used[st[tp--]]=0;
                }
                used[i]=1;
                st[++tp]=i;
            }
        }
        for(int i=1;i<=tp;i++){
            convexhull.push_back(p[st[i]]);
        }
    }
    double getPerimeter(){
        double ans=0;
        for(int i=1;i<convexhull.size();i++){
            ans+=convexhull[i].dis(convexhull[i-1]);
        }
        return ans;
    }
    double getArea(){
        if(convexhull.size()<4) return 0;
        double ans=0;
        for(int i=1;i<convexhull.size()-2;i++){
            ans+=(convexhull[i]-convexhull[0])^(convexhull[i+1]-convexhull[0])/2;
        }
        return ans;
    }
    //旋转卡壳求直径
    double getLongest(){
        if(convexhull.size()<4){
            return convexhull[0].dis(convexhull[1]);
        }
        int j=0;
        const int sz=convexhull.size();
        double ans=0;
        for(int i=0;i<convexhull.size()-1;i++){
            Line line(convexhull[i],convexhull[i+1],1);
            while(line.distancetopoint(convexhull[j])<=line.distancetopoint(convexhull[(j+1)%sz])){
                j=(j+1)%sz;
            }
            ans=max({ans,(convexhull[i]-convexhull[j]).norm(),(convexhull[i+1]-convexhull[j]).norm()});
        }
        return ans;
    }
    //旋转卡壳最小矩形覆盖
    pair<double,vector<Point>> minRectangleCover(){
        vector<Point> p;
        if(convexhull.size()<4) return {0,p};
        int j=1,l=1,r=1;
        double ans=1e18;
        const int sz=convexhull.size();
        for(int i=1;i<convexhull.size();i++){
            Line line(convexhull[i-1],convexhull[i],1);
            while(line.distancetopoint(convexhull[j])<=line.distancetopoint(convexhull[(j+1)%sz])){
                j=(j+1)%sz;
            }
            while((convexhull[i]-convexhull[i-1])*(convexhull[(r+1)%sz]-convexhull[i-1])>=(convexhull[i]-convexhull[i-1])*(convexhull[r]-convexhull[i-1])){
                r=(r+1)%sz;
            }
            if(i==1) l=r;
            while((convexhull[i-1]-convexhull[i])*(convexhull[(l+1)%sz]-convexhull[i])>=(convexhull[i-1]-convexhull[i])*(convexhull[l]-convexhull[i])){
                l=(l+1)%sz;
            }
            Point t1=convexhull[i]-convexhull[i-1];
            Point t2=convexhull[r]-convexhull[i];
            Point t3=convexhull[l]-convexhull[i-1];
            double a=line.distancetopoint(convexhull[j]);
            double b=t1.norm()+t1*t2/t1.norm()-t1*t3/t1.norm();
            double tmp=a*b;
            if(ans>tmp){
                ans=tmp;
                p.clear();
                p.push_back(t1*((t1*t3)/(t1.norm()*t1.norm()))+convexhull[i-1]);
                p.push_back(t1*(1+(t1*t2)/(t1.norm()*t1.norm()))+convexhull[i-1]);
                Point tmp=Point{-(p[1]-p[0]).y,(p[1]-p[0]).x}*a/b;
                p.push_back(tmp+p[1]);
                p.push_back(tmp+p[0]);
            }
        }
        return {ans,p};
    }
};
```

## 博弈论

### SG函数

SG(x)，x是游戏的状态，SG=0，先手必败，否则先手必胜

设后继状态为a1,a2,……ap，SG(x)=mex(SG(a1),SG(a2)……)

一个游戏的SG函数值等于各个游戏SG函数值的nim和（异或和）

## dp

### 数位dp

给定区间[l,r]，问区间满足条件的数有多少个，cal(r)-cal(l-1)

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



## 随机算法

### 模拟退火

$T$：温度

$△ T$：温度变化率，每次温度等于上一次T*△T，一般取0.95-0.99，模拟徐徐降温

$x$：当前选择的解

$△x$：解变动量

$x1$：当前的目标解，等于x+△x

$△f$：当前解的函数值与目标解函数值的差值，等于$f(x)-f(x1)$

每次的$△x$在一个大小与T成正比的值域内随机取值。如果$f(x1)<f(x)$，那么接受目标解x=x1，如果$f(x1)>f(x)$，则以一定的概率接受，概率是$e^{\frac{-△f}{T}}$，直到T趋近于0，循环结束

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

#### 拉链法哈希

```cpp
cc_hash_table <int,int> f;
```

#### 红黑树

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

insert(x)：向树中插入一个元素 x，返回 std::pair<point_iterator, bool>。
erase(x)：从树中删除一个元素/迭代器 x，返回一个 bool 表明是否删除成功。
order_of_key(x)：返回 x 以 Cmp_Fn 比较的排名，**0开始**。
find_by_order(x)：返回 Cmp_Fn 比较的排名所对应元素的迭代器。
lower_bound(x)：以 Cmp_Fn 比较做 lower_bound，返回迭代器。
upper_bound(x)：以 Cmp_Fn 比较做 upper_bound，返回迭代器。
join(x)：将 x 树并入当前树，前提是两棵树的类型一样，x 树被删除。
split(x,b)：以 Cmp_Fn 比较，小于等于 x 的属于当前树，其余的属于 b 树。
empty()：返回是否为空。
size()：返回大小。

元素不可重，可以使用pair<int,int>加入时间戳来使元素可重

```cpp
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

