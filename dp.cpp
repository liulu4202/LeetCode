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

