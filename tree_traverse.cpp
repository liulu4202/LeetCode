#include <iostream>
using namespace std;
struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void preOrder1(TreeNode* root){
  if(root){
    cout << root->val << " ";
    preOrder(root->left);
    preOrder(root->right);
  }
}

void preOrder2(TreeNode* root){
  stack<TreeNode*> st;
  while(!st.empty()||root!=nullptr){
    if(root){
      cout << root->val << " ";
      st.push(root);
      root = root->left;
    }else{
      root = st.top();
      st.pop();
      root = root->right;
    }
  }
}

void inOrder1(TreeNode* root){
  if(root){
    inOrder(root->left);
    cout << root->val << " ";
    inOrder(root->right);
  }
}

void inOrder2(TreeNode* root){
  stack<TreeNode*> st;
  while(!st.empty()||root!=nullptr){
    while(root!=nullptr){
      st.push(root);
      root = root->left;
    }
    root = st.top();
    st.pop();
    cout << root->val << " ";
    root = root->right;
  }
}

void postOrder1(TreeNode* root){
  if(root){
    postOrder1(root->left);
    cout << root->val << " ";
    postOrder1(root->right);
  }
}

void postOrder2(TreeNode* root){
  stack<TreeNode*> st;
  TreeNode* nodeLast = root;//需要判断上次访问的是左子树还是右子树
  while(!st.empty()||root!=nullptr){
    while(root!=nullptr){
      st.push(root);
      root = root->left;
    }
    root = st.top();
    if(root->right==nullptr||nodeLast==root->right){//判断当前节点是否有右子节点，若有则判断是否已经遍历过，若已经遍历过则遍历当前节点
      cout << root->val << " ";
      st.pop();
      nodeLast = root;
      root = nullptr;
    }else{
      root = root->right;
    }
  }
}

void floorOrder(TreeNode* root){
  queue<TreeNode*> q;
  if(root!=nullptr) q.push(root);
  while(!q.empty()){
    root = q.front();
    cout << root->val << " ";
    if(root->left!=nullptr)
        q.push(root->left);
    if(root->right!=nullptr)
        q.push(root->right);
    q.pop();
  }
}

/*116. 填充每个节点的下一个右侧节点指针
方法一：层序遍历
方法二：使用已建立的next指针
*/
class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL) return root;
        queue<Node*> q;
        q.push(root);
        while(!q.empty()){
            int size = q.size();
            for(int i = 0;i < size;i++){
                Node* node = q.front();
                q.pop();
                if(i < size - 1){
                    node->next = q.front();
                }
                if(node->left!=NULL) q.push(node->left);
                if(node->right!=NULL) q.push(node->right);
            }
        }
        return root;
    }
};

class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL) return root;
        
        Node* leftmost = root;
        while(leftmost->left!=NULL){
            Node* head = leftmost;
            while(head!=NULL){
                head->left->next = head->right;
                if(head->next!=NULL){
                    head->right->next = head->next->left;
                }
                head = head->next;
            }
            leftmost = leftmost->left;
        }
        return root;
    }
};
//使用已建立的next指针的递归实现
class Solution {
public:
    Node* connect(Node* root) {
        if(root == NULL) return root;
        if(root->left!=NULL){
            root->left->next = root->right;
            root->right->next = (root->next!=NULL) ? root->next->left : NULL;
            connect(root->left);
            connect(root->right);
        }
        return root;
    }
};

/*117. 填充每个节点的下一个右侧节点指针 II
方法一：层序遍历 方法与116一样
方法二：使用已建立的next指针*/

class Solution {
public:
    void helper(Node* &last, Node* &p, Node* &nextStart){
        if(last) last->next = p;
        if(!nextStart) nextStart = p;
        last = p;
    }
    Node* connect(Node* root) {
        if(root == NULL) return root;
        Node* node = root;
        while(node){
            Node *last = NULL, *nextStart = NULL;
            for(Node* p = node;p != NULL;p=p->next){
                if(p->left) helper(last, p->left, nextStart);
                if(p->right) helper(last, p->right, nextStart);
            }
            node = nextStart;
        }
        return root;
    }
};

/*124. 二叉树中的最大路径和*/
class Solution {
private:
    int maxSum = INT_MIN;
public:
    int pathSum(TreeNode* root) {
        if(root == nullptr) return 0;
        int leftSum = max(pathSum(root->left), 0);
        int rightSum = max(pathSum(root->right), 0);
        int sumNew = root->val + leftSum + rightSum;
        maxSum = max(maxSum, sumNew);
        return root->val + max(leftSum, rightSum);
    }
    
    int maxPathSum(TreeNode* root) {
        pathSum(root);
        return maxSum;
    }
};

/*剑指 Offer 33. 二叉搜索树的后序遍历序列
方法一：递归 根据二叉搜索树的定义，可以通过递归，判断所有子树的 正确性 即其后序遍历是否满足二叉搜索树的定义），若所有子树都正确，则此序列为二叉搜索树的后序遍历。
*/
class Solution {
public:
    bool verify(vector<int>& postorder, int l, int r){
        if(l >= r) return true;
        int i = l;
        while(postorder[i] < postorder[r]) i++;
        int idx = i;
        while(postorder[i] > postorder[r]) i++;
        return i==r && verify(postorder, l, idx - 1) && verify(postorder, idx, r - 1);

    }
    bool verifyPostorder(vector<int>& postorder) {
        int n = postorder.size();
        return verify(postorder, 0, n - 1);
    }
};

/*236. 二叉树的最近公共祖先
递归：最近的公共祖先满足((l && r) || ((root == p || root == q) && (l || r)))，更远的公共祖先必有l或r为false，无法满足*/
class Solution {
public:
    TreeNode* ans;
    bool dfs(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == NULL) return false;
        bool l = dfs(root->left, p, q);
        bool r = dfs(root->right, p, q);
        if((l && r) || ((root == p || root == q) && (l || r)))
            ans = root;
        return l || r || (root == p || root == q);
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        dfs(root, p, q);
        return ans;
    }
};

/*572. 另一个树的子树*/
class Solution {
public:
    bool check(TreeNode* s, TreeNode* t){
        if(!s && !t) return true;
        if((!s && t) || (s && !t) || (s->val != t->val)) return false;
        return check(s->left, t->left) && check(s->right, t->right);
    }
    // 对s的每棵子树都要深入到s的每个节点进行判断
    bool dfs(TreeNode* s, TreeNode* t){
        if(!s) return false;
        return check(s,t) || dfs(s->left, t) || dfs(s->right, t);
    }
    bool isSubtree(TreeNode* s, TreeNode* t) {
        return dfs(s, t);
    }
};
