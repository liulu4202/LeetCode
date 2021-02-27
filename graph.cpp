/*207. 课程表
方法一：DFS 对于图中的任意一个节点，它在搜索的过程中有三种状态，即：
「未搜索」：我们还没有搜索到这个节点；
「搜索中」：我们搜索过这个节点，但还没有回溯到该节点，即该节点还没有入栈，还有相邻的节点没有搜索完成）；
「已完成」：我们搜索过并且回溯过这个节点，即该节点已经入栈，并且所有该节点的相邻节点都出现在栈的更底部的位置，满足拓扑排序的要求。
通过上述的三种状态，我们就可以给出使用深度优先搜索得到拓扑排序的算法流程，在每一轮的搜索搜索开始时，我们任取一个「未搜索」的节点开始进行深度优先搜索。
我们将当前搜索的节点u标记为「搜索中」，遍历该节点的每一个相邻节点v：
如果v为「未搜索」，那么我们开始搜索v，待搜索完成回溯到u；
如果v为「搜索中」，那么我们就找到了图中的一个环，因此是不存在拓扑排序的；
如果v为「已完成」，那么说明v已经在栈中了，而u还不在栈中，因此u无论何时入栈都不会影响到(u,v)之前的拓扑关系，以及不用进行任何操作。
当u的所有相邻节点都为「已完成」时，我们将u放入栈中，并将其标记为「已完成」。
在整个深度优先搜索的过程结束后，如果我们没有找到图中的环，那么栈中存储这所有的 nn 个节点，从栈顶到栈底的顺序即为一种拓扑排序。
方法二：bfs
考虑拓扑排序中最前面的节点，该节点一定不会有任何入边，也就是它没有任何的先修课程要求。当我们将一个节点加入答案中后，我们就可以移除它的所有出边，代表着它的相邻节点少了一门先修课程的要求。如果某个相邻节点变成了「没有任何入边的节点」，那么就代表着这门课可以开始学习了。按照这样的流程，我们不断地将没有入边的节点加入答案，直到答案中包含所有的节点（得到了一种拓扑排序）或者不存在没有入边的节点（图中包含环）。
我们使用一个队列来进行广度优先搜索。初始时，所有入度为0的节点都被放入队列中，它们就是可以作为拓扑排序最前面的节点，并且它们之间的相对顺序是无关紧要的。
在广度优先搜索的每一步中，我们取出队首的节点u：
我们将u放入答案中；我们移除u的所有出边，也就是将u的所有相邻节点的入度减少1。如果某个相邻节点v的入度变为0，那么我们就将v放入队列中。
*/
class Solution {
private:
    vector<vector<int>> edges;
    vector<int> visited;
    bool valid = true;

public:
    void dfs(int u){
        visited[u] = 1;
        for(auto v: edges[u]){
            if(visited[v] == 0){
                dfs(v);
                if(!valid) return; 
            }else if(visited[v] == 1){
                valid = false;
                return;
            }
        }
        visited[u] = 2;
    }

    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        edges.resize(numCourses);
        visited.resize(numCourses);
        for(const auto& info: prerequisites){
            edges[info[1]].push_back(info[0]);
        }
        for(int i = 0;i < numCourses&&valid;i++){
            if(!visited[i]){
                dfs(i);
            }
        }
        return valid;    
    }
};

class Solution {
private:
    vector<vector<int>> edges;
    vector<int> indeg;
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        edges.resize(numCourses);
        indeg.resize(numCourses);
        for(const auto& info: prerequisites){
            edges[info[1]].push_back(info[0]);
            ++indeg[info[0]];
        }
        queue<int> q;
        for(int i = 0;i < numCourses;i++){
            if(indeg[i] == 0){
                q.push(i);
            }
        }
        int visited = 0;
        while(!q.empty()){
            visited++;
            int u = q.front();
            q.pop();
            for(int v:edges[u]){
                --indeg[v];
                if(indeg[v] == 0){
                    q.push(v);
                }
            }
        }
        return visited == numCourses;    
    }
};

/*210. 课程表 II 与201基本一致，但是题目要求返回学习顺序，所以需要一个栈来保存拓扑顺序
*/
class Solution {
private:
    vector<vector<int>> edges;
    vector<int> visited;
    vector<int> path;
    bool valid = true;

public:
    void dfs(int u){
        visited[u] = 1;
        for(auto v: edges[u]){
            if(visited[v] == 0){
                dfs(v);
                if(!valid) return; 
            }else if(visited[v] == 1){
                valid = false;
                return;
            }
        }
        visited[u] = 2;
        path.push_back(u);
    }

    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        edges.resize(numCourses);
        visited.resize(numCourses);
        for(const auto& info: prerequisites){
            edges[info[1]].push_back(info[0]);
        }
        for(int i = 0;i < numCourses&&valid;i++){
            if(!visited[i]){
                dfs(i);
            }
        }
        if (!valid) {
            return {};
        }
        reverse(path.begin(), path.end());
        return path;    
    }
};

class Solution {
private:
    vector<vector<int>> edges;
    vector<int> indeg;
    vector<int> path;
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        edges.resize(numCourses);
        indeg.resize(numCourses);
        for(const auto& info: prerequisites){
            edges[info[1]].push_back(info[0]);
            ++indeg[info[0]];
        }
        queue<int> q;
        for(int i = 0;i < numCourses;i++){
            if(indeg[i] == 0){
                q.push(i);
            }
        }
        int visited = 0;
        while(!q.empty()){
            visited++;
            int u = q.front();
            q.pop();
            path.push_back(u);
            for(int v:edges[u]){
                --indeg[v];
                if(indeg[v] == 0){
                    q.push(v);
                }
            }
        }
        if(visited != numCourses) return {};
        return path;    
    }
};

/*990. 等式方程的可满足性
并查集：如果!=相连的两个变量属于同一个父节点则返回false，否则为true
*/
class UnionFind{
private:
    vector<int> parent;
public:
    UnionFind(){
        parent.resize(26);
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int index){
        if(index == parent[index]){
            return index;
        }
        parent[index] = find(parent[index]);
        return parent[index];
    }

    void united(int index1, int index2){
        parent[find(index1)] = find(index2);
    }
};

class Solution {
public:
    bool equationsPossible(vector<string>& equations) {
        UnionFind uf;
        for(const auto& s:equations){
            if(s[1] == '='){
                int index1 = s[0] - 'a';
                int index2 = s[3] - 'a';
                uf.united(index1, index2);
            }
        }
        for(const auto& s:equations){
            if(s[1] == '!'){
                int index1 = s[0] - 'a';
                int index2 = s[3] - 'a';
                if(uf.find(index1) == uf.find(index2)){
                    return false;
                }
            }
        }
        return true;
    }
};