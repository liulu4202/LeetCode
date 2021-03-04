/*5. 最长回文子串 给你一个字符串s，找到s中最长的回文子串。
方法一：DP，用dp记录s[i]和s[j]之间是否为回文，同时记录最优长度与起点
方法二：中心扩展算法 
*/
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        int maxLen = 1, start = 0;
        for(int i = 0;i < n;i++){
            dp[i][i] = 1;
            if(i < n - 1 && s[i + 1] == s[i]){
                dp[i][i + 1] = 1;
                maxLen = 2;
                start = i;
            }
        }
        for(int l = 3;l <= n;l++){
            for(int i = 0; i < n - l + 1;i++){
                int j = i + l - 1;
                if(s[i] == s[j] && dp[i+1][j-1] == 1){
                    dp[i][j] = dp[i+1][j-1];
                    maxLen = l;
                    start = i;
                } 
            }
        }
        return s.substr(start, maxLen);
    }
};

class Solution {
public:
    string longestPalindrome(string s) {
        int left = 0, right = 0;
        int n = s.size();
        for(int i = 0;i < n;i++){
            if(2*(n-i)+1 < right - left + 1) break;
            int l = i, r = i;//奇数长度的子串
            while(l >= 0 && r < n && s[l] == s[r]){
                l--;
                r++;
            }
            if(r - l - 2 > right - left){
                left = l + 1;
                right = r - 1;
            }
            l = i;r = i + 1;//偶数长度的子串
            while(l >= 0 && r < n && s[l] == s[r]){
                l--;
                r++;
            }
            if(r - l - 2 > right - left){
                left = l + 1;
                right = r - 1;
            }
        } 
        return s.substr(left, right - left + 1);
    }
};

/*516. 最长回文子序列
用dp[i][j]表示s[i]~s[j]之间回文字符串的最大长度，为了保证求解时使用的子问题是求解过的，第一层循环为i--
*/
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));

        for(int i = n - 1;i >= 0;i--){
            dp[i][i] = 1;
            for(int j = i + 1;j < n;j++){
                if(s[i] == s[j]){
                    dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 2);
                }
                else{
                    dp[i][j] = max(dp[i][j], max(dp[i+1][j], dp[i][j-1]));
                }
            }
        }
        return dp[0][n-1];
    }
};

/*72. 编辑距离
删除A中一个字符等价于在B中插入一个字符，所以实际操作只有替换和插入
用dp[i][j]记录word1[0:i]和word2[0:j]的编辑距离，如果word1[i]=word2[j]，则dp[i][j]=dp[i-1][j-1]
否则dp[i][j]=min(dp[i-1][j-1]+1(在word1中替换), dp[i-1][j]+1(在word1中插入), dp[i][j-1]+1(在word2中插入))
*/
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        if(m * n == 0) return m+n;
        vector<vector<int>> dp(m+1, vector<int>(n+1));
        for(int i = 0;i <= m;i++){
            dp[i][0] = i;
        }
        for(int j = 0;j <= n;j++){
            dp[0][j] = j;
        }
        for(int i = 1;i <= m;i++){
            for(int j = 1;j <= n;j++){
                if(word1[i-1] == word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = min(dp[i-1][j-1]+1, min(dp[i][j-1]+1, dp[i-1][j]+1));
                }
            }
        }
        return dp[m][n];
    }
};

/*1143. 最长公共子序列

*/
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n =text2.size();
        if(m * n == 0) return 0;
        vector<vector<int>> dp(m+1, vector<int>(n+1));
        for(int i = 0;i <= m;i++) dp[i][0] = 0;
        for(int j = 0;j <= n;j++) dp[0][j] = 0;
        for(int i = 1;i <= m;i++){
            for(int j = 1;j <= n;j++){
                if(text1[i-1] == text2[j-1]){
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else{
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
};



class LongestSubstring {
public:
    int findLongest(string A, int n, string B, int m) {
        // write code here
        if(m*n == 0) return 0;
        vector<vector<int>> dp(n+1, vector<int>(m+1));
        int ans = 0;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                if(A[i-1] == B[j-1]){
                    dp[i][j] = dp[i-1][j-1] + 1;
                    if(dp[i][j] > ans){
                        ans = dp[i][j];
                    }
                }else{
                    dp[i][j] = 0;
                }
            }
        }
        return ans;
    }
};


/*376. 摆动序列
方法一：dp，摆动可以拆分成【上升摆动序列：最后一个元素呈上升趋势】和【下降摆动序列：最后一个元素呈下降趋势】
分别用up[i]和down[i]表示前i个元素所能构成的最长上升摆动序列和下降摆动序列，初始条件up[0]=down[0]=1; 
则if(nums[i]<=nums[i-1]) up[i]=up[i-1],if(nums[i]>nums[i-1]) up[i]=max(up[i-1],down[i-1]+1),down[i]则相反，
结果为max(up[n-1], down[n-1])
方法二：优化的dp 由于当前状态的判断只需要前一个状态，所以可以变量up，down代替dp数组
*/
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        if(n < 2) return n;
        vector<int> up(n);
        vector<int> down(n);
        up[0] = 1; down[0] = 1;
        for(int i = 1;i < n;i++){
            if(nums[i] < nums[i-1]){
                up[i] = up[i-1];
                down[i] = max(down[i-1], up[i-1]+1);
            }else if(nums[i] > nums[i-1]){
                up[i] = max(up[i-1], down[i-1]+1);
                down[i] = down[i-1];
            }else{
                up[i] = up[i-1];
                down[i] = down[i-1];
            }
        }
        return max(up[n-1],down[n-1]);
    }
};

class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        if(n < 2) return n;
        int up = 1, down = 1, ans = 1;
        for(int i = 1;i < n;i++){
            if(nums[i] < nums[i-1]){
                down[i] = max(down, up+1);
            }else if(nums[i] > nums[i-1]){
                up = max(up, down[i-1]+1);
            }
        }
        return max(up,down);
    }
};

/*300. 最长递增子序列
方法一：DP，一开始被摆动序列误导了，以为该题和摆动序列一样，实际是不同的；要求最长递增子序列，那么需要根据max(dp[j]), 0<=j<i&nums[j]<nums[i]确定dp[i]的大小
方法二：贪心+二分查找 考虑一个简单的贪心，如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升子序列最后加上的那个数尽可能的小。
维护一个数组d[i]，表示长度为i的最长上升子序列的末尾元素的最小值，用len记录目前最长上升子序列的长度，起始时len为1，d[1]=nums[0]。
d为单调递增数组，可以用二分法查找查找d[i-1]<nums[i]<d[i] ，
依次遍历数组nums中的每个元素，并更新数组d和len的值。如果nums[i]>d[len]则更新len++，否则在d[1…len]中找满足d[i−1]<nums[j]<d[i]的下标i，并更新d[i]=nums[j]
*/

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        if(n < 2) return n;
        int ans = 0;
        vector<int> dp(n);
        for(int i = 0;i < n;i++){
            int maxLen = 1;
            for(int j = 0;j < i;j++){
                if(nums[i] > nums[j]){
                    maxLen = max(maxLen, dp[j]+1);
                }
            }
            dp[i] = maxLen;
            ans = max(ans, maxLen);
        }
        return ans;
    }
};

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = 1, n = (int)nums.size();
        if (n == 0) {
            return 0;
        }
        vector<int> d(n + 1, 0);
        d[len] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0;  // 如果找不到说明所有的数都比 nums[i] 大，此时要更新 d[1]，所以这里将 pos 设为 0
                while (l <= r) {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i]) {   // 目标：找到比nums[i]小的最大的数
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
                // 找到第一个大于等于nums[i]的数，进行替换
                d[pos + 1] = nums[i];
            }
        }
        return len;
    }
};

//牛客网：'最长上升子序列' 实战变种：搭积木
/*小明有一袋子长方形的积木，如果一个积木A的长和宽都不大于另外一个积木B的长和宽，则积木A可以搭在积木B的上面。
好奇的小明特别想知道这一袋子积木最多可以搭多少层，你能帮他想想办法吗？
贪心+二分查找 用d[i]记录层数为i的积木的最下面一层积木的长宽最大值，由于存在宽度相同的积木，d[i]中存在重复数字所以二分查找是在包含重复数字的数组中寻找第一个大于等于
woods[i].second的值
*/
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std; 

static bool cmp(const pair<int, int> w1, const pair<int, int> w2){
    if(w1.first == w2.first){
        return w1.second < w2.second;
    }
    return w1.first < w2.first;
}

int stacked_wood(int n, vector<pair<int, int>>& woods){
    if(n <= 1) return n;
    vector<int> d(n+1, 0);
    sort(woods.begin(), woods.end(), cmp);
    int len = 1;
    d[len] = woods[0].second;
    for(int i = 1; i < n;i++){
        if(d[len] <= woods[i].second){
            d[++len] = woods[i].second;
        }else{
            int l = 1, r = len;
            while(l < r){
                int mid = (l + r) >> 1;
                if(woods[i].second <= d[mid]){
                    r = mid;
                }else{
                    l = mid + 1;
                }
            }
            d[l] = woods[i].second;
        }
    }
    return len;
}

int main(){
    vector<pair<int, int>> woods;
    int n;
    cin >> n;
    for(int i = 0;i < n;i++){
        int f1, f2;
        cin >> f1 >> f2;
        woods.push_back({f1, f2});
    }
    cout << stacked_wood(n, woods);
};

/*674. 最长连续递增序列
用滑动窗口记录递增子序列
*/
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int n = nums.size();
        if(n <= 1) return n;
        int start = 0, ans = 1;
        for(int i = 1;i < n;i++){
            if(nums[i] > nums[i-1]){
                ans = max(ans, i - start + 1);
            }else{
                start = i;
            }
        }
        return ans;
    }
};

/*673. 最长递增子序列的个数
找到满足nums[j]<nums[i]的最大的j，这样的j有多个，根据dp[j]+1与dp[i]的大小关系判断是遇到的几个j
*/
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        if(n <= 1) return n;
        vector<int> cnt(n, 1), dp(n, 1);
        
        for(int i = 1;i < n;i++){
            for(int j = 0;j < i;j++){
                if(nums[i] > nums[j]){
                    if(dp[j] + 1 > dp[i]){
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j]; 
                    }else if(dp[j] + 1 == dp[i]){
                        cnt[i] += cnt[j];
                    }
                }
            }
        }
        int maxLen = *max_element(dp.begin(), dp.end());
        int res = 0;
        for(int i = 0;i < n;i++){
            if(dp[i] == maxLen) res += cnt[i];
        }
        
        return res;
    }
};

/*128. 最长连续序列 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
进阶：你可以设计并实现时间复杂度为 O(n) 的解决方案吗？
用set保存数组内元素，以O(1)复杂度进行查找
*/
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> s;
        for(auto n: nums){
            s.insert(n);
        }
        int ans = 0;
        for(auto n: nums){
	    if(s.count(n - 1)!=0) continue;
            int cur = 1, curNum = n;
            while(s.count(curNum+1)!=0){
                curNum++;
                cur++;
            }
            ans = (ans>cur?ans:cur);
        }
        return ans;
    }
};

/*10. 正则表达式匹配 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素 所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
*/
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        if(m != 0 & n == 0) return false;
        vector<vector<int>> dp(m+1, vector<int>(n+1));
        dp[0][0] = 1;
        auto match = [&s, &p](int i, int j) {
            if(i == 0)
                return false;
            if(p[j - 1] == '.')
                return true;
            return s[i - 1] == p[j - 1];
        };

        for(int i = 0;i <= m;i++){
            for(int j = 1;j <= n;j++){
                if(p[j - 1] == '*'){
                    if(match(i, j - 1)){
                        dp[i][j] |= dp[i-1][j];
                    }
                    if(j >= 2){
                        dp[i][j] |= dp[i][j-2];
                    }
                }else{
                    if(match(i, j)) 
                        dp[i][j] |= dp[i - 1][j - 1];
                }
            } 
        }
        return dp[m][n];
    }
};

/*44. 通配符匹配
*/
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        if(p == "*") return true;
        vector<vector<int>> dp(m+1, vector<int>(n+1));
        dp[0][0] = 1;
        for(int j = 1;j <= n;j++){
            if(p[j-1] == '*'){
                dp[0][j] = 1;
            }else{
                break;
            }
        }
        
        for(int i = 1;i <= m;i++){
            for(int j = 1;j <= n;j++){
                if(p[j-1] == s[i-1] || p[j-1] == '?'){
                    dp[i][j] = dp[i-1][j-1];
                }else if(p[j-1] == '*'){
                    dp[i][j] = dp[i][j-1]||dp[i-1][j];
                }else{
                    dp[i][j] = 0;
                }
            }
        }
        return dp[m][n];
    }
};

/*53. 最大子序和 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
dp[i]=max(dp[i-1]+nums[i], nums[i])
*/
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int ans = nums[0], sum = nums[0];
        for(int i = 1;i < n;i++){
            sum = max(nums[i], sum + nums[i]);
            ans = max(ans, sum);
        }
        
        return ans;
    }
};

/*152. 乘积最大子数组
dp 数组中存在负值，所以需要记录最大和最小值，以便负负得正
由于当前状态只和前一个状态有关，所以根据「滚动数组」思想，我们可以只用两个变量来维护i−1时刻的状态
*/
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        vector<int> maxF(nums), minF(nums);
        for(int i = 1;i < nums.size();i++){
            maxF[i] = max(maxF[i - 1] * nums[i], max(minF[i - 1] * nums[i], nums[i]));
            minF[i] = min(maxF[i - 1] * nums[i], min(minF[i - 1] * nums[i], nums[i]));
        }
        return *max_element(maxF.begin(), maxF.end());
    }
};

class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int maxF = nums[0], minF = nums[0], ans = nums[0];
        for(int i = 1;i < nums.size();i++){
            int mx = maxF, mn = minF;
            maxF = max(mx*nums[i], max(mn*nums[i], nums[i]));
            minF = min(mx*nums[i], min(mn*nums[i], nums[i]));
            ans = max(ans, maxF);
        }
        return ans;
    }
};

/*120. 三角形最小路径和
方法一：dp f[i][j]=min(f[i-1][j-1], f[i-1][j])+c[i][j]
方法二：由于f[i][j]只与前两个状态有关，因此我们不必存储这些无关的状态。具体地，我们使用两个长度为n的一维数组进行转移，
将i根据奇偶性映射到其中一个一维数组，那么i−1就映射到了另一个一维数组。这样我们使用这两个一维数组，交替地进行状态转移
f[j]=min(f[j-1],f[j])+c[i][j]
*/
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        if(n == 0) return 0;
        vector<vector<int>> dp(n, vector<int>(n, 0));
        dp[0][0] = triangle[0][0];
        for(int i = 1;i < n;i++){
            dp[i][0] = dp[i - 1][0] + triangle[i][0];
            for(int j = 1;j < i;j++){
                dp[i][j] = triangle[i][j] + min(dp[i - 1][j], dp[i - 1][j - 1]);
            }
            dp[i][i] = dp[i - 1][i - 1] + triangle[i][i];
        }
        
        return *min_element(dp[n - 1].begin(), dp[n - 1].end());
    }
};

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        if(n == 0) return 0;
        vector<int> f(n);
        f[0] = triangle[0][0];
        for(int i = 1;i < n;i++){
            f[i] = f[i - 1] + triangle[i][i];
            for(int j = i - 1;j > 0;j--){
                f[j] = triangle[i][j] + min(f[j], f[j - 1]);
            }
            f[0] += triangle[i][0];
        }
        
        return *min_element(f.begin(), f.end());
    }
};


/*85. 最大矩形
*/
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if(m == 0) return 0;
        int n = matrix[0].size();
        if(n == 0) return 0;
        int ans = 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                if(matrix[i][j] == '1')
                    dp[i][j] = (j == 0) ? 1 : dp[i][j - 1] + 1;
                int width = dp[i][j];
                for(int k = i;k >= 0;k--){
                    width = min(width, dp[k][j]);
                    ans = max(ans, width*(i - k + 1));
                }
            }
        }
        return ans;
    }
};

/*221. 最大正方形
dp(i,j)=min(dp(i−1,j),dp(i−1,j−1),dp(i,j−1))+1
*/
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if(m == 0) return 0;
        int n = matrix[0].size();
        if(n == 0) return 0;
        int maxSide = 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                if(matrix[i][j] == '1'){
                    if(i == 0||j == 0)
                        dp[i][j] = 1;
                    else
                        dp[i][j] = min(dp[i-1][j-1], min(dp[i][j-1], dp[i-1][j])) + 1;
                }
                maxSide = max(dp[i][j], maxSide);
            }
        }
        return maxSide * maxSide;
    }
};

/*32. 最长有效括号
方法一：dp，s[i]=‘)’且s[i−1]=‘(’，也就是字符串形如“……()”，可以推出dp[i]=dp[i−2]+2;s[i]=‘)’且s[i−1]=‘)’，也就是字符串形如“……))”，
可以推出:如果s[i−dp[i−1]−1]=‘(’，dp[i]=dp[i−1]+dp[i−dp[i−1]−2]+2
方法二：栈
对于遇到的每个‘(’ ，我们将它的下标放入栈中；
对于遇到的每个‘)’，我们先弹出栈顶元素表示匹配了当前右括号：
如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
*/
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.size(), ans = 0;
        if(n == 0) return 0;
        vector<int> dp(n, 0);
        for(int i = 1;i < n;i++){
            if(s[i] == ')'){
                if(s[i - 1] == '('){
                    dp[i] = (i >= 2) ? dp[i - 2] + 2 : 2;
                }else if(i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '('){
                    dp[i] = dp[i-1] + (i - dp[i - 1] >= 2) ? dp[i-dp[i-1]-2]:0 + 2;
                }
                ans = max(ans, dp[i]);
            }
        }
        return ans;
    }
};

class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0;
        stack<int> stk;
        stk.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                stk.push(i);
            } else {
                stk.pop();
                if (stk.empty()) {
                    stk.push(i);
                } else {
                    maxans = max(maxans, i - stk.top());
                }
            }
        }
        return maxans;
    }
};

class Solution {
public:
    int longestValidParentheses(string s) {
        int left = 0, right = 0, maxlength = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = max(maxlength, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }
        left = right = 0;
        for (int i = (int)s.length() - 1; i >= 0; i--) {
            if (s[i] == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = max(maxlength, 2 * left);
            } else if (left > right) {
                left = right = 0;
            }
        }
        return maxlength;
    }
};

/*91. 解码方法
*/
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        if(n == 0 || s[0] == '0') return 0;
        vector<int> dp(n, 0);
        dp[0] = 1; 
        for(int i = 1;i < n;i++){
            if(s[i] == '0') {
                if(s[i-1] == '1' || s[i-1] == '2')
                    dp[i] = ((i >= 2) ? dp[i - 2]:1);
                else return 0;
            }else if(s[i-1] == '1' || (s[i-1] == '2' && s[i]>='1' && s[i]<='6')){
                dp[i] = dp[i - 1] + ((i >= 2) ? dp[i - 2]:1);
            }else{
                dp[i] = dp[i - 1];
            }
        }
        return dp[n - 1];
    }
};

class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        if(n == 0 || s[0] == '0') return 0;
        int pre = 1, curr = 1;
        for (int i = 1; i < s.size(); i++) {
            int tmp = curr;
            if (s[i] == '0')
                if (s[i - 1] == '1' || s[i - 1] == '2') curr = pre;
                else return 0;
            else if (s[i - 1] == '1' || (s[i - 1] == '2' && s[i] <= '6'))
                curr += pre;
            pre = tmp;
        }
        return curr;
    }
};

/*剑指 Offer 46. 把数字翻译成字符串
*/
class Solution {
public:
    int translateNum(int num) {
        if(num == 0) return 1;
        int pre = 1, cur = 1;
        int lastNum = num % 10, curNum;
        num /= 10;
        while(num > 0){
            curNum = num % 10;
            int tmp = cur;
            if(curNum == 1 || (curNum == 2 && lastNum <= 5))
                cur += pre;
            pre = tmp;
            lastNum = curNum;
            num /= 10;
        }
        return cur;
    }
};

/*121. 买卖股票的最佳时机
*/
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int buy = -prices[0], sell = 0;
        for(int i = 0;i < prices.size();i++){
            buy = max(buy, -prices[i]);
            sell = max(sell, buy + prices[i]);
        }
        return sell;
    }
};

/*122. 买卖股票的最佳时机 II
方法一：动态规划：当前有两种状态，持有股票和未持有股票，分别用两个变量记录
方法二：贪心：ans=∑max{0,a[i]−a[i−1]}
*/
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int dp[n][2];
        dp[0][0] = 0, dp[0][1] = -prices[0];
        for (int i = 1; i < n; ++i) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }
};

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int dp0 = 0, dp1 = -prices[0];
        for(int i = 1;i < n;i++){
            int newdp0 = max(dp0, dp1 + prices[i]);
            int newdp1 = max(dp1, dp0 - prices[i]);
            dp0 = newdp0;
            dp1 = newdp1;
        }
        return dp0;
    }
};

/*123. 买卖股票的最佳时机 III
*/
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int buy1 = -prices[0], sell1 = 0, buy2 = -prices[0], sell2 = 0;
        for(int i=1;i<prices.size();i++){
            buy1 = max(buy1, -prices[i]);
            sell1 = max(sell1, buy1+prices[i]);
            buy2 = max(buy2, sell1-prices[i]);
            sell2 = max(sell2, buy2+prices[i]);
        }
        return sell2;
    }
};

/*198. 打家劫舍
*/
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) return 0;
        if(n == 1) return nums[0]; 
        vector<int> dp(n, 0);
        dp[0] = nums[0]; dp[1] = max(nums[0], nums[1]);
        for(int i = 2;i < n;i++){
            dp[i] = max(dp[i - 1], dp[i - 2]+nums[i]);
        }
        return dp[n - 1];
    }
};

/*213. 打家劫舍 II
和198相比，增加了环状结构，可以截成两部分进行动态规划，取其中较大的作为结果*/
class Solution {
public:
    int dp(vector<int> nums, int start, int end){
        int first = nums[start], second = max(nums[start], nums[start + 1]), tmp;
        for(int i = start + 2;i < end;i++){
            tmp = second;
            second = max(second, first + nums[i]);
            first = tmp;

        }
        return second;
    }

    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) return 0;
        if(n == 1) return nums[0]; 
        if(n > 1 && n <= 3) return *max_element(nums.begin(), nums.end());
        
        return max(dp(nums, 0, n - 1), dp(nums, 1, n));
    }

};

/*279. 完全平方数
*/
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n + 1, 0);
        for(int i = 1;i <= n;i++){
            dp[i] = i;
            for(int j = 1;i - j * j >= 0;j++){
                dp[i] = min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }
};

/*746. 使用最小花费爬楼梯
*/
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n, 0);
        dp[0] = cost[0];
        dp[1] = cost[1];
        for(int i = 2;i < n;i++){
            dp[i] = min(dp[i - 2], dp[i - 1]) + cost[i];
        }
        return min(dp[n - 1], dp[n - 2]);
    }
};

/*63. 不同路径 II
*/
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        if(m < 1) return 0;
        int n = obstacleGrid[0].size();
        if(n < 1) return 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for(int i = 0;i < m;i++){
            if(obstacleGrid[i][0] == 1) break;
            dp[i][0] = 1;
        }
        for(int j = 0;j < n;j++){
            if(obstacleGrid[0][j] == 1) break;
            dp[0][j] = 1;
        }
            
        for(int i = 1;i < m;i++){
            for(int j = 1;j < n;j++){
                if(obstacleGrid[i][j] == 0)
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                else
                    dp[i][j] = 0;
            }
        }
        return dp[m - 1][n - 1];
    }
};

/*322. 零钱兑换
*/
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<int> dp(amount+1, amount+1);
        dp[0] = 0;
        for(int i = 0;i < n;i++){
            for(int j = coins[i];j <= amount;j++){
                dp[j] = min(dp[j], dp[j - coins[i]] + 1);
            }
        }
        return (dp[amount] == amount + 1) ? -1 : dp[amount];
    }
};

/*983. 最低票价
*/
class Solution {
private:
    unordered_set<int> dayset;
    vector<int> memo;
public:
    int mincostTickets(vector<int>& days, vector<int>& costs) {
        memo.resize(366, -1); memo[0] = 0;
        for(auto x: days) dayset.insert(x);
        return dp(1, costs);
    }

    int dp(int start, vector<int>& costs) {
        if(start > 365) return 0;
        if(memo[start] >= 0) return memo[start];
        if(dayset.count(start)) 
            memo[start] = min(dp(start + 1, costs) + costs[0], 
                        min(dp(start + 7, costs) + costs[1],
                        dp(start + 30, costs) + costs[2]));
        else 
            memo[start] = dp(start + 1, costs);
        return memo[start];
    }
};
