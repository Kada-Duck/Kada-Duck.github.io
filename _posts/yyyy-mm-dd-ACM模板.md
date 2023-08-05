---
layout:     post
title:      (文章标题)
subtitle:   (副标题)
date:       2018-12-01
author:     (作者名)
header-img: img/the-first.png
catalog:   true
tags:
    - 学习资料
---

# ACM模板



### 头文件

```c++
#include <iostream>
#include <cstdio>
#include <queue>
#include <set>
#include <map>
#include <cmath>
#include <climits>
#include <algorithm>
#include <stack>
#include <cstring>
#define low(x) ((x) & (-(x) ) )
#define E(x) ((x) * (x))
#define ma make_pair
#define rep(a,b,c) for(int a=b;a<=c;a++)
#define per(a,b,c) for(int a=b;a>=c;a--)
#define inf 0x3f3f3f3f
using namespace std;
typedef long long LL;

char ch1;
template<class T>
inline void rd(T& x) {
	x = 0; bool w = 0;
	ch1 = getchar();
	while (!isdigit(ch1)) { ch1 == '-' && (w = 1), ch1 = getchar(); }
	while (isdigit(ch1)) { x = (x << 1) + (x << 3) + (ch1 & 15), ch1 = getchar(); }
	w && (x = (~x) + 1);
}
template<class T>
inline void wr(T x)
{
	if (x < 0) x = -x, putchar('-');
	if (x < 10) {
		putchar(x + 48);
		return;
	}
	T L = x / 10;
	wr(L);
	putchar(x - ((L << 1) + (L << 3)) + 48);
}
/*int head[N],tot;
struct edge{
	int to,nxt;
}e[M];

void add(int u,int v){
	e[++tot].to = v;
	e[tot].nxt = head[u];
	head[u] = tot;
}*/

bool cp(int a,int b){return a > b;} // 大到小

int gcd(int a,int b){
	if(b == 0)return a;
	return gcd(b,a % b);
}
int T;

int main(){
	return 0;
}
```

##### python

```python
import math as m
t = eval(input())
for _ in range(t):
	n = int(input())
	a = list(map(int , input().split() ) )
```



### 线段树

#### 支持[l,r]乘,加，求和 

```c++
const int N = 100005;
int n,m,s,t,mod; 
LL a[N],data[N << 2],k;
int size[N << 2],lazy_a[N << 2],lazy_m[N << 2];

void update(int x){
	data[x] = (data[x << 1] + data[x << 1 | 1]) % mod;
}

void ad(int x,LL v){ // 值，+ 
	data[x] = (data[x] + size[x] * v % mod) % mod;		
	lazy_a[x] = (lazy_a[x] + v) % mod;
}

void mul(int x,LL v){ // 更新值,+,* 
	data[x] = data[x] * v % mod;
	lazy_m[x] = lazy_m[x] * v % mod;
	lazy_a[x] = lazy_a[x] * v % mod;	
}

void pd(int x){
	mul(x << 1,lazy_m[x]);
	mul(x << 1 | 1,lazy_m[x]);
	ad(x << 1,lazy_a[x]);
	ad(x << 1 | 1,lazy_a[x]);
	lazy_a[x] = 0;
	lazy_m[x] = 1;
}

void build(int x,int l,int r){
	if(l == r){
		data[x] = a[l];
		size[x] = 1;
		return;
	}
	lazy_m[x] = 1;
	build(x << 1,l,mid);
	build(x << 1 | 1,mid + 1,r);
	size[x] = size[x << 1] + size[x << 1 | 1];
	update(x); 
}

void modify(int x,int l,int r){
	if(s <= l && r <= t){
		ad(x,k);
		return;
	}
	pd(x);
	if(s <= mid)modify(x << 1,l,mid);
	if(t > mid)modify(x << 1 | 1,mid + 1,r);
	update(x);
}

void multi(int x,int l,int r){
	if(s <= l && r <= t){
		mul(x,k);
		return;
	}
	pd(x);
	if(s <= mid)multi(x << 1,l,mid);
	if(t > mid)multi(x << 1 | 1,mid + 1,r);
	update(x);
}

int query(int x,int l,int r){
	if(s <= l && r <= t)return data[x];
	int res = 0;
	pd(x);
	if(s <= mid)res = query(x << 1,l,mid);
	if(t > mid)res = (res + query(x << 1 | 1,mid + 1,r)) % mod;
	return res;
}

//[l,r]乘,加，求和 
int main(){
	rd(n),rd(m),rd(mod);
	rep(i,1,n){
		rd(a[i]);
		a[i] %= mod;
	}
	build(1,1,n); 
	while(m--){
		int op;
		rd(op);
		if(op == 2){
			rd(s),rd(t),rd(k);
			modify(1,1,n);	
		}
		else {
			if(op == 1){
				rd(s),rd(t),rd(k);
				multi(1,1,n);	
			}
			else {
				rd(s),rd(t);
				wr(query(1,1,n));
				cout << '\n';	
			}
		}
	}
	return 0;
}
```



### BIT

```c++
// 区修，区查BIT
//#define low(x) (x & (-x))
LL a[100010],b[100010],p,q,k;
int n,m,s,t;

LL sum(int x){LL ans = 0;for(int i = x;i;i -= low(i))ans += (x + 1) * a[i] - b[i];return ans;} // (x + 1)套差分数组 
LL query(int l,int r){return sum(r) - sum(l - 1);}
void ad(int x,LL k){for(int i = x;i <= n;i += low(i))a[i] += k,b[i] += k * x;} // b[i]  i * d[i]的差分数组 
void modify(int l,int r,LL k){ad(l,k),ad(r + 1,-k);}

int main(){
	rd(n),rd(m);
	p = 0;
	rep(i,1,n){
		rd(q);
		ad(i,q - p);
		p = q; 
	}
	while(m --){
		int op;
		rd(op);
		if(op == 1){
			rd(s),rd(t),rd(k);
			modify(s,t,k);
		}
		else {
			rd(s),rd(t);
			wr(query(s,t));
			cout << '\n';
		}
	}
	return 0;
}
```



### exgcd

```c++
// ax + by = d
LL exgcd(LL a, LL b, LL &x, LL &y){
    if(b == 0){
        x = 1, y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```



### 二分

```c++
int l,r,mid;
while(l < r){ //尽可能小 
    mid = l + r >> 1;
    if(check(mid))r = mid;
    else l = mid + 1;
}

int l,r,mid;
while(l < r){ // 大
    mid = (l + r + 1) >> 1;
    if(check(mid))l = mid;
    else r = mid - 1;
}

```

### 矩乘

```c++
struct node{
	int g[6][6];
}a,f;

void mul(node &x,node y,node z){
	memset(x.g, 0 ,sizeof x.g);
	rep(i,0,5)rep(j,0,5)rep(k,0,5)x.g[i][k] = x.g[i][k] + 1ll * y.g[i][j] * z.g[j][k]; 
}
int main(){
	while(k){
		if(k & 1)mul(f,f,a);
		k >>= 1;
		mul(a,a,a);
	}
}
```







```c++
int n, m, ans, fa[maxN * 3];

int find(int u) { return fa[u] == u ? u : fa[u] = find(fa[u]); }

int main() {
	n = read(), m = read();
	for (int i = 1; i <= n * 3; i++) { fa[i] = i; }
	for (; m; m--) {
		int opt = read(), u = read(), v = read();
		if (u > n || v > n) { ans++; continue; }
		if (opt == 1) {
			if (find(u + n) == find(v) || find(u) == find(v + n)) { ans++; }
			else {
				fa[find(u)] = find(v);
				fa[find(u + n)] = find(v + n);
				fa[find(u + n + n)] = find(v + n + n);
			}
		} else {
			if (find(u) == find(v) || find(u) == find(v + n)) { ans++; }
			else {
				fa[find(u + n)] = find(v);
				fa[find(u + n + n)] = find(v + n);
				fa[find(u)] = find(v + n + n);
			}
		}
	}
	printf("%d\n", ans);
	return 0;
}
```

## 最短路

#### 朴素dij

```c++
const int N = 2010;
int n,m,K,T,t[N],dis[N],vis[N];
struct edge{
	int v,k;
};
vector<edge>g[N];

void dij(){
	rep(k,1,n){
		int mn = inf,x = 0;
		rep(i,1,n)if(vis[i] == 0 && dis[i] < mn)x = i,mn = dis[i];
		vis[x] = 1;
		for(auto ed : g[x]){
			if(dis[ed.v] == inf)continue;
			dis[ed.k] = min(dis[ed.k] , max(dis[x],dis[ed.v]) + max(t[x],t[ed.v]) );
		}
	}
}

int main(){
	rd(n),rd(m),rd(K),rd(T);
	rep(i,1,n)rd(t[i]);
	
	memset(dis,0x3f , sizeof dis);
	rep(i,1,m){
		int x;
		rd(x),dis[x] = 0; // 种子时间 
	}
	rep(i,1,K){
		int a,b,c;
		rd(a),rd(b),rd(c);
		g[a].push_back({b,c});
		g[b].push_back({a,c});
	}
	dij();
	wr(dis[T]);
	return 0;
}
```

#### 优先队列优化dij

```c++
const int N = 1e5 + 7;

std::priority_queue<pa >q;
int n,m,s;
int to[N<<1],last[N<<1],val[N<<1];
int head[N],tot;
int dist[N];
bool vis[N];

void addedge(int u,int v, int w){
    to[++tot] = v,last[tot] = head[u],val[tot] = w,head[u] = tot;
}

int main(){
    scanf("%d %d %d",&n,&m,&s);
    while(m--){
        int u,v,w;
        scanf("%d %d %d",&u,&v,&w);
        addedge(u,v,w);
    }
    for(int i = 1;i <= n;i++)dist[i] = INT_MAX;
    dist[1] = 0;
    q.push(ma(0,1));
    while(!q.empty()){
        int x = q.top().second;
        q.pop();
        if(vis[x] == 1)continue;
        vis[x] = 1;
        for(int i = head[x];i;i = last[i]){
            int v = to[i],w = val[i];
            if(dist[v] > w + dist[x]){
                dist[v] = w + dist[x];
                q.push(ma(-dist[v],v));
            }
        }
    }
    for(int i = 1;i <= n;i++){
        printf("%d ",dist[i]);
    }
    return 0;
}
```



#### SPFA

```c++
const int N = 2010;
int n,m,K,T,t[N],dis[N],vis[N];
struct edge{
	int v,k;
};
vector<edge>g[N];

queue<int>q;

void spfa(){
	while(!q.empty()){
		int x = q.front();
		q.pop();
		vis[x] = 0;
		for(auto ed : g[x]){
			if(dis[ed.v] == inf)continue;
			if(dis[ed.k] > max(dis[x],dis[ed.v]) + max(t[x],t[ed.v]) )
			{
				dis[ed.k] = max(dis[x],dis[ed.v]) + max(t[x],t[ed.v]) ;
				if(!vis[ed.k])vis[ed.k] = 1,q.push(ed.k);
			}
		}
	}
}

int main(){
	rd(n),rd(m),rd(K),rd(T);
	rep(i,1,n)rd(t[i]);
	
	memset(dis,0x3f , sizeof dis);
	rep(i,1,m){
		int x;
		rd(x),dis[x] = 0; // 种子时间 
		q.push(x);
		vis[x] = 1;
	}
	rep(i,1,K){
		int a,b,c;
		rd(a),rd(b),rd(c);
		g[a].push_back({b,c});
		g[b].push_back({a,c});
	}
	spfa();
	wr(dis[T]);
	return 0;
}
```



#### 找负环

```c++
struct edge {
  int v, w;
};

vector<edge> e[maxn];
int dis[maxn], cnt[maxn], vis[maxn];
queue<int> q;

bool spfa(int n, int s) {
	memset(dis, 63, sizeof(dis));
	dis[s] = 0, vis[s] = 1;
	q.push(s);
	while (!q.empty()) {
    	int u = q.front();
    	q.pop(), vis[u] = 0;
    	for (auto ed : e[u]) {
    		int v = ed.v, w = ed.w;
    		if (dis[v] > dis[u] + w) {
        		dis[v] = dis[u] + w;
        		cnt[v] = cnt[u] + 1;  // 记录最短路经过的边数
    			if (cnt[v] >= n) return false;
        		if (!vis[v]) q.push(v), vis[v] = 1;
      		}
    	}
	}
	return true;
}

```

