/*88. 合并两个有序数组
解法同合并两个有序链表*/ 
/**********************************************双指针*****************************************************/
/*76. 最小覆盖子串
方法：双指针+辅助空间记录字符串t是否被完全覆盖*/
class Solution {
public:
    string minWindow(string s, string t) {
        vector<int> need(128,0);
        int count = 0;  
        for(char c : t)
        {
            need[c]++;
        }
        count = t.length();
        int l=0, r=0, start=0, size = INT_MAX;
        while(r<s.length())
        {
            char c = s[r];
            if(need[c]>0)
                count--;
            need[c]--;  //先把右边的字符加入窗口
            if(count==0)    //窗口中已经包含所需的全部字符
            {
                while(l<r && need[s[l]]<0) //缩减窗口
                {
                    need[s[l++]]++;
                }   //此时窗口符合要求
                if(r-l+1 < size)    //更新答案
                {
                    size = r-l+1;
                    start = l;
                }
                need[s[l]]++;   //左边界右移之前需要释放need[s[l]]
                l++;
                count++;
            }
            r++;
        }
        return size==INT_MAX ? "" : s.substr(start, size);
    }
};

/*26. 删除排序数组中的重复项
方法：双指针，头指针记录当前不重复的最后一位，尾指针逐个遍历数组，判断当前值是否与前一个值相等*/
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) return 0;
        int i = 0;
        for(int j=0;j<n;j++){
            if(nums[j] != nums[i]){
                nums[++i] = nums[j];
            }
        }
        return i+1;
    }
};

/*283. 移动零
方法：双指针，用头指针记录当前处理过的0值的最后一位，尾指针遍历每个元素*/
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) return ;

        int i = 0;
        for(int j = 0;j < n;j++){
            if(nums[j] != 0){
                swap(nums[i++], nums[j]);
            }
        }
    }
};

/*75. 颜色分类
双指针，头指针标记0的最后一位，尾指针标记2的第一位*/
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            //如果交换之后的num[i]仍然是2，那么继续交换，直到nums[i]为1，否则2有可能被交换到头部位置，而对应索引已经遍历完就无法再调整
            while (i <= p2 && nums[i] == 2) {
                swap(nums[i], nums[p2]);
                --p2;
            }
            if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                ++p0;
            }
        }
    }
};

/*11. 盛最多水的容器
双指针，左右指针各代表当前左右边界，移动较小的边界，计算移动后的结果*/
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size()-1, ans = 0;
        while(left < right){
            int t = (height[left]<height[right]?height[left]:height[right])*(right - left);
            ans = (ans>t?ans:t);
            if(height[left]<height[right]){
                left++;
            }else{
                right--;
            }
        }
        return ans;
    }
};


/*42. 接雨水
方法一：dp，leftMax[i]，rightMax[i]记录每一个height[i]对应的左右侧高度
方法二：单调栈
方法三：双指针，类似11题，移动较小的边界，维护当前最大左右边界leftMax和rightMax，根据边界高度计算水量*/
class Solution {
public:
  int trap2(vector<int>& height) {
    int n = height.size(), ans = 0;
    if(n <= 1)
      return 0;
    vector<int> leftMax(n, 0), rightMax(n, 0);
    leftMax[0] = height[0];
    for(int i = 1; i < n; i ++) {
      leftMax[i] = max(height[i], leftMax[i-1]);
    }
    rightMax[n-1] = height[n-1];
    for(int i = n-2; i >= 0; i --) {
      rightMax[i] = max(height[i], rightMax[i+1]);
    }
    for(int i = 1; i <= n-2; i ++) {
      ans += min(leftMax[i], rightMax[i]) - height[i];
    }
    return ans;
  }


class Solution {
public:
    int trap(vector<int>& height) {
        stack<int> s;
        int ans = 0;
        for(int i = 0; i < height.size(); i++){
            //单调递减栈
            while(!s.empty() && height[i] > height[s.top()]){
                int top = s.top();
                s.pop();
                if(s.empty()) break;
                int distance = i - s.top() - 1;
                int bounded_height = min(height[i], height[s.top()]) - height[top];
                ans += bounded_height*distance;
            }
            s.push(i);
        }
        return ans;
    }
};

int trap(vector<int>& height)
{
    int left = 0, right = height.size() - 1;
    int ans = 0;
    int left_max = 0, right_max = 0;
    while (left < right) {
        if (height[left] < height[right]) {
            height[left] >= left_max ? (left_max = height[left]) : ans += (left_max - height[left]);
            ++left;
        }
        else {
            height[right] >= right_max ? (right_max = height[right]) : ans += (right_max - height[right]);
            --right;
        }
    }
    return ans;
}
/***********************************************二分查找*******************************************************/
/**牛课网：有序矩阵查找 行、列均有序，则从右上角开始搜索*/
class Finder {
public:
    bool findX(vector<vector<int> > mat, int n, int m, int x) {
        // write code here
        int i = 0, j = m - 1;
        while(i < n && j >= 0){
            if(x > mat[i][j]){
                ++ i;
            } else if(x < mat[i][j]){
                -- j;
            } else {
                return true;  // return {i, j};
            }
        }
        return false; // return {};
    }
};

/*33. 搜索旋转排序数组
首先根绝nums[mid]和nums[0]、nums[-1]的大小关系确定mid在第一段还是第二段，如在第一段则根据target和nums[mid],nums[0]的大小确定边界移动，
否则根据target和nums[mid],nums[-1]的大小确定边界移动 */
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.size()==0) return -1;
        int n = nums.size();
        int left = 0, right = n - 1;
        while(left<=right){
            int mid = left + (right - left)/2;
            if(target == nums[mid]) return mid;
            if(nums[mid] >= nums[0]){//在第一段
                if(target < nums[mid] && target >= nums[0]){
                    right = mid-1;
                }else{
                    left = mid+1;
                }
            }else{
                if(target > nums[mid] && target<=nums[n-1]){
                    left = mid+1;
                }else{
                    right = mid-1;
                }
            }
        }
        return -1;
    }
};

/*189. 旋转数组
方法一：环状替换， 用count记录替换的次数
方法二：反转，这个方法基于这个事实：当我们旋转数组k次， k%n个尾部元素会被移动到头部，剩下的元素会被向后移动*/
class Solution {
public:
  void rotate(vector<int>& nums, int k) {
    int n = nums.size(), count = 0;
    k %= n;
    for(int i = 0; count < n; i ++) {
      int c = i, prev = nums[i];
      do {
        int j = (c + k) % n;
        int curr = nums[j];
        nums[j] = prev;
        prev = curr;
        c = j;
        count ++;
      } while(c != i);
    }
  }
};

class Solution {
public:
  void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k %= n;
    reverse(nums, 0, n - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, n - 1);
  }

  void reverse(vector<int>& nums, int start, int end) {
    while(start < end) {
      swap(nums[start ++], nums[end --]);
    }
  }
};
/***************************************************横纵索引反转数组*******************************************************/
/*剑指 Offer 29. 顺时针打印矩阵*/
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ans;
        if(matrix.size()==0 || matrix[0].size()==0) return ans;
        int m = matrix.size(), n = matrix[0].size();
        int l = 0, r = n-1, u = 0, d = m-1;
        while(true){
            for(int i = l;i <= r;i++){
                ans.push_back(matrix[u][i]);
            }
            if(++u>d) break;
            for(int i = u;i<=d;i++){
                ans.push_back(matrix[i][r]);
            }
            if(--r<l) break;
            for(int i = r;i>=l;i--){
                ans.push_back(matrix[d][i]);
            }
            if(--d<u) break;
            for(int i = d;i>=u;i--){
                ans.push_back(matrix[i][l]);
            }
            if(++l>r) break;
        }
        return ans;
    }
};

/*48. 旋转图像
方法一：原地旋转，根绝索引关系matrix[i][j] = matrix[n - j - 1][i]推导
方法二：用翻转代替旋转*/
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        if(matrix.size()==0) return;
        int n = matrix.size();
        for(int i = 0;i < n / 2;i++){
            for(int j = 0;j < (n + 1) / 2;j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j -1];
                matrix[n - i - 1][n - j -1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = tmp;
            }
        }
    }
};

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                swap(matrix[i][j], matrix[n - i - 1][j]);
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};

/*15. 三数之和
*/
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        if(nums.size()<3) return {};
        int n = nums.size();
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        for(int first = 0;first < n;first++){
            if(nums[first]>0) return ans;
            if(first>0 && nums[first] == nums[first - 1]) continue;
            int second = first + 1, third = n - 1, target = - nums[first];
            while(second < third){
                if(nums[second] + nums[third] == target){
                    ans.push_back({nums[first], nums[second], nums[third]});
                    while(second < third && nums[second] == nums[second + 1])
                        second++;
                    while(second < third && nums[third] == nums[third - 1])
                        third--;
                    second++;
                    third--;
                }else if(nums[second] + nums[third] < target){
                    second++;
                }else{
                    third--;
                }
            }
        }
        return ans;
    }
};

/*16. 最接近的三数之和*/
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int ans = nums[0]+nums[1]+nums[2];
        for(int first = 0;first < n;first++){
            int second = first + 1, third = n - 1; 
            while(second < third){
                int sum = nums[first]+nums[second]+nums[third];
                if(abs(target - ans)>abs(target - sum)) ans = sum;
                if(sum == target) return sum;
                else if(sum < target){
                    second++;
                }else{
                    third--;
                }
            }
        }
        return ans;
    }
};

/*18. 四数之和*/
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int n = nums.size();
        vector<vector<int>> ans;
        if(n < 4) return ans;
        sort(nums.begin(), nums.end());
        for(int first = 0;first < n - 3;first++){
            if(first > 0 && nums[first] == nums[first - 1]) continue;
            for(int second = first + 1;second < n - 2;second++){
                if(second > first + 1 && nums[second] == nums[second - 1]) continue;
                int third = second + 1, fourth = n - 1;
                int new_target = target - nums[first] - nums[second];

                while(third < fourth){
                    if(nums[third] + nums[fourth] == new_target){
                        ans.push_back({nums[first],nums[second],nums[third],nums[fourth]});
                        while(third < fourth && nums[third] == nums[third + 1]){
                            third++;
                        }
                        while(third < fourth && nums[fourth] == nums[fourth - 1]){
                            fourth--;
                        }
                        third++;
                        fourth--;
                    }else if(nums[third] + nums[fourth] < new_target){
                        third++;
                    }else{
                        fourth--;
                    }
                }
            }
        }
        return ans;
    }
};
/********************************************************排列组合***************************************************************/
/*78. 子集 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。*/
class Solution {
public:
    vector<int> t;
    vector<vector<int>> ans;

    void dfs(int cur, vector<int>& nums) {
        if (cur == nums.size()) {
            ans.push_back(t);
            return;
        }
        t.push_back(nums[cur]);
        dfs(cur + 1, nums);//使用nums[cur]
        t.pop_back();
        dfs(cur + 1, nums);//不使用nums[cur]
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0, nums);
        return ans;
    }
};

/*46. 全排列 给定一个 没有重复 数字的序列，返回其所有可能的全排列。*/
class Solution {
public:
    void backtrack(vector<vector<int>>& res, vector<int>& output, int first, int len){
        // 所有数都填完了
        if (first == len) {
            res.emplace_back(output);
            return;
        }
        for (int i = first; i < len; ++i) {
            // 动态维护数组
            swap(output[i], output[first]);//代表当前位置（索引first处）填入nums[i]
            // 继续递归填下一个数
            backtrack(res, output, first + 1, len);
            // 撤销操作
            swap(output[i], output[first]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int> > res;
        backtrack(res, nums, 0, (int)nums.size());
        return res;
    }
};

/*47. 全排列 II
给定一个可包含重复数字的序列 nums，按任意顺序 返回所有不重复的全排列。*/
class Solution {
public:
    vector<int> visits;
    void backtrace(vector<vector<int>>& ans, vector<int>& path, vector<int>& nums, int idx){
        if(idx == nums.size()){
            ans.emplace_back(path);
            return;
        }
        for(int i = 0;i < nums.size();i++){
            if(visits[i]||(i > 0 && nums[i] == nums[i - 1] && visits[i - 1])) continue;//用visits标记保证每个位置仅有重复元素的第一个填入
            path.emplace_back(nums[i]);
            visits[i] = 1;
            backtrace(ans, path, nums, idx + 1);
            visits[i] = 0;
            path.pop_back();
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> path;
        visits.resize(nums.size());
        sort(nums.begin(), nums.end());
        backtrace(ans, path, nums, 0);
        return ans;
    }
};

/*31. 下一个排列*/
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        int i = n - 2;
        while(i >= 0 && nums[i]>=nums[i+1]) i--;//单调递减序列代表最大值，因此要从右侧找第一个非降序的元素，即最右侧的a[i]<a[i+1]
        if(i>=0){
            int j = n - 1;
            while(j >= i && nums[j]<=nums[i]) j--;//a[i+1:n-1]为单调减序列，找出第一个大于a[i]的元素a[j]，交换a[i]和a[j]
            swap(nums[i], nums[j]);
        }
        
        int head = i + 1, tail = n - 1;
        while(head < tail){//反转a[i+1:n-1]为单调升序
            swap(nums[head], nums[tail]);
            head++;
            tail--;
        }
    }
};

/*
39. 组合总和
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的数字可以无限制重复被选取。
说明：
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 
*/
class Solution {
private:
  vector< vector<int> > ans;
public:
  vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end());
    backTracking(candidates, {}, 0, target);
    return ans;
  }

  void backTracking(vector<int>& candidates, vector<int> perm, int idx, int target) {//回溯
    if(target == 0) {
      ans.push_back(perm); return;
    }
    for(int i = idx; i < candidates.size(); i ++) {
      if(target - candidates[i] < 0)//剪枝
        break;
      perm.push_back(candidates[i]);
      backTracking(candidates, perm, i, target - candidates[i]);
      perm.pop_back();
    }
  }
};

/*40. 组合总和 II
给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的每个数字在每个组合中只能使用一次。
说明：
所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 
*/
class Solution {
private:
vector<vector<int>> ans;
vector<int> visited;
public:
  vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    visited.resize(candidates.size(), 0);
    sort(candidates.begin(), candidates.end());
    backTracking(candidates, {}, target, 0);
    return ans;
  }

  void backTracking(vector<int>& candidates, vector<int> perm, int target, int idx) {
    if(target == 0) {
      ans.push_back(perm); return;
    }
    for(int i = idx; i < candidates.size(); i ++) {
      if(target - candidates[i] < 0) break; 
      if(i > idx && candidates[i] == candidates[i - 1] && !visited[i - 1]) continue;
      if(!visited[i]) {//使用visited进行层间剪枝，耗时从8ms降低到4ms，也可以不用
        visited[i] = 1;
        perm.push_back(candidates[i]);
        backTracking(candidates, perm, target - candidates[i], i + 1);
        perm.pop_back();
        visited[i] = 0;
      }
    }
  } 
};
/**********************************************************原地重判*************************************************************/
/*41. 缺失的第一个正数 给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数
从1开始枚举正整数，用哈希表查询该数字是否存在；为了减小空间复杂度，借助nums构造一个哈希表，将<1和>n的值都修改成n+1，然后遍历数组中的每一个数x，
它可能已经被打了标记，因此原本对应的数为abs(x)。如果x>=1&&x<=N+1，那么给数组中的nums[x-1]打标记（取负值），nums[x-1]已经是负值无需再改，然后遍历数组，
如果数组中的每一个数都是负数，那么答案是N+1，否则答案是第一个正数的位置加1。
*/
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for(int i = 0;i < n;i++){
            if(nums[i] <= 0) nums[i] = n + 1;
        }
        for(int i = 0;i < n;i++){
            int num = abs(nums[i]);
            if(num <= n) {
                nums[num - 1] = -abs(nums[num - 1]);
            }
        }
        for(int i = 0;i < n;i++){
            if(nums[i] > 0) return i + 1;
        }
        return n + 1;
    }
};



/*******************************************前缀和*******************************************/
/*287. 寻找重复数 
给定一个包含n+1个整数的数组nums，其数字都在1到n之间（包括1和n），可知至少存在一个重复的整数。假设nums只有一个重复的整数 ，找出这个重复的数 。
方法一：通过构造有序数组实现二分查找。用cnt[i]记录小于数字i数字个数，如果重复数字为j，那么对于任意i<j有cnt[i]<=i;对于i>=j，有cnt[i]>i;cnt为有序数组
为了减少空间复杂度，循环计算cnt值代替数组；
方法二：快慢指针：每个位置i->nums[i]，则对于重复元素一定有至少两条边指向该元素，整张图一定构成环；我们先设置慢指针slow和快指针fast，
慢指针每次走一步，快指针每次走两步，根据「Floyd 判圈算法」两个指针在有环的情况下一定会相遇，此时我们再将slow放置起点0，两个指针每次同时移动一步，
相遇的点就是答案。
方法三：原地重判，类似41题，如果如果nums[abs(nums[i]) - 1]>0则证明为abs(nums[i])第一次出现，取负值；如果nums[abs(nums[i]) - 1]<0则说明abs(nums[i])为第二次出现
*/
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        int left = 1, right = n - 1, ans = -1;
        while(left <= right){
          int mid = (left + right) >> 1;
          int cnt = 0;
          for(int i = 0;i < n;i++){
            cnt += nums[i]<=mid;
          }
          if(cnt<=mid) left = mid + 1;
          else {
            right = mid - 1;
            ans = mid;
          }
        }
        return ans;
    }
};

class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};

class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        for(int i = 0;i < n;i++){
            int x = abs(nums[i]);
            if(nums[x - 1] < 0) return x;
            else nums[x - 1] = -nums[x - 1];
        }
        return -1;
    }
};

/*209. 长度最小的子数组 给定一个含有n个正整数的数组和一个正整数s，找出该数组中满足其和≥s的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回0。
方法一：前缀和+二分查找 先用一个数组sum[i]来保存从nums[0]到nums[i]的和，nums中元素为正整数，所以sum为递增序列，可以使用二分查找求解
方法二：双指针，右移右指针扩展窗口，维护变量sum存储的子数组和，如果sum>=s，则右移左指针，更新子数组长度；
*/
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        int ans = INT_MAX;
        vector<int> sums(n + 1, 0); 
        // 为了方便计算，令 size = n + 1 
        // sums[0] = 0 意味着前 0 个元素的前缀和为 0
        // sums[1] = A[0] 前 1 个元素的前缀和为 A[0]
        // 以此类推
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 1; i <= n; i++) {
            int target = s + sums[i - 1];
            auto bound = lower_bound(sums.begin(), sums.end(), target);
            if (bound != sums.end()) {
                ans = min(ans, static_cast<int>((bound - sums.begin()) - (i - 1)));
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};

class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        if(nums.size() == 0) return 0;
        int n = nums.size();
        int start = 0, end = 0, sum = 0;
        int ans = INT_MAX;
        while(end < n){
            sum += nums[end];
            while(sum >= target){
                ans = min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == INT_MAX? 0:ans;
    }
};

/*：LintCode 139: 最接近零的子数组和*/
/*
做法就是利用前缀和，先用一个数组sum[i]来保存从nums[0]到nums[i]的和，同时还要记录下标。
那么，我们想要得到nums[i]到nums[j]的和，只要用sum[j] - sum[i-1]就可以了。
剩下的工作就是对sum数组排序，找到排序后相邻的差的绝对值最小的那一对节点。
*/
class Solution {
public:
    /*
     * @param nums: A list of integers
     * @return: A list of integers includes the index of the first number and the index of the last number
     */
    vector<int> subarraySumClosest(vector<int> &nums) {
        // write your code here
        int n = nums.size();
        vector< pair<int, int> > sum(n + 1);
        sum[0].first = 0; sum[0].second = -1;
        for(int i = 0; i < n; i ++) {
            sum[i + 1].first = sum[i].first + nums[i];
            sum[i + 1].second = i;
        }
        sort(sum.begin(), sum.end());
        vector<int> ans(2);
        int diff = INT_MAX;
        for(int i = 1; i <= n; i ++) {
            int curr = abs(sum[i].first - sum[i - 1].first);
            if(curr < diff) {
                diff = curr;
                ans[0] = min(sum[i].second, sum[i - 1].second) + 1;
                ans[1] = max(sum[i].second, sum[i - 1].second);
            }
        }
        return ans;
    }
};

/*166. 分数到小数*/
/*
给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。
如果小数部分为循环小数，则将循环的部分括在括号内。
如果存在多个答案，只需返回 任意一个 。
对于所有给定的输入，保证 答案字符串的长度小于 104 。
*/
// 这道题目的关键用到一个数学定理：一个分数不是有限小数，就是无限循环小数
class Solution {
public:
    string fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }

        string res;
        bool isNegative = false;
        if (numerator > 0 && denominator < 0 || numerator < 0 && denominator > 0) {
            isNegative = true;
        }

        long num = labs(numerator);
        long denom = labs(denominator);
        long remainder = num;
        // 如果分子大于分母
        if (num >= denom) {
            remainder = num % denom;
            // 大于0的部分
            res += to_string(num / denom);
        }
        // 是不是整除
        if (remainder == 0) {
            return isNegative ? "-" + res : res;
        }

        // 如果分子小于分母，就是0开头
        if (remainder == labs(num)) {
            res += "0";
        }
        // 加小数点开始小数部分
        res += ".";
        unordered_map<int, int> rem2Pos;
        int index = res.size();

       // 分数一定是有限小数或者无限循环小数，这里通过 remainder=0 或者 remainder重复出现 作为循环退出的判断条件
        while (remainder && !rem2Pos.count(remainder)) {
            rem2Pos[remainder] = index++;
            // 技巧，分数除法向后借10
            remainder *= 10;
            int digit = remainder / denom;
            remainder = remainder % denom;
            res += to_string(digit);
        }

        if (rem2Pos.count(remainder)) {
          // 上面记录index是为了下面插入'('，循环小数的开始位置
            res.insert(rem2Pos[remainder], "(");
            res.push_back(')');
        }

        return isNegative ? "-" + res : res;
    }
};

/*842. 将数组拆分成斐波那契序列
回溯+剪枝，注意剪枝3个剪枝条件 1.大于0的数字不能以0开头；2.数字不能超过INT_MAX；3.当前两个数字已经确定时当前值不能大于前两个数字之和
*/
class Solution {
public:
    bool dfs(vector<int>& res, string s, int len, int start, long long sum, long long prev){
        if(start==len) {
            return res.size() >= 3;
        }

        long long cur = 0;
        for(int i=start;i<len;i++){
            if(i>start && s[start]=='0') {
                break;
            }
            cur = cur*10+s[i]-'0';
            if(cur>INT_MAX){
                break;
            }
            if(res.size()>=2){
                if(cur<sum) {
                    continue;
                }else if(cur>sum){
                    break;
                }
            }
            res.push_back(cur);
            if(dfs(res,s,len,i+1,prev+cur,cur)) {
                return true;
            }
            res.pop_back();
        }
        return false;
    }

    vector<int> splitIntoFibonacci(string S) {
        vector<int> res;
        dfs(res,S,S.size(),0,0,0);
        return res;
    }
};

/*剑指 Offer 14- I. 剪绳子
最开始的想法是回溯，但是会有许多重复计算，由于该问题可以分解子问题，且具有最优子结构性质，因此可以使用dp解决，用dp[i]存放i所能切割出的最大乘积，
dp[i] = max(dp[i], max(j * dp[i - j], j * (i - j)))
*/
class Solution {
public:
    int cuttingRope(int n) {
        vector<int> dp(n + 1);
        dp[1] = 1;dp[2] = 1;
        for(int i = 3;i < n + 1;i++){
            for(int j = 2;j < i;j++){
                dp[i] = max(dp[i], max(j * dp[i - j], j * (i - j)));
            }
        }
        return dp[n];
    }
};

/*剑指 Offer 14- II. 剪绳子
贪心：为使乘积最大，只有长度为2和3的绳子不应再切分，且3比2更优。（推导几个值即可看出）
*/
class Solution {
public:
    int cuttingRope(int n) {
        if (n <= 3) 
            return n - 1;
        long rs = 1;
        while (n > 4) {
            rs *= 3;
            rs %= 1000000007;
            n -= 3;
        }
        return (rs * n) % 1000000007;
    }
};

/*219. 存在重复元素 II 给定一个整数数组和一个整数k，判断数组中是否存在两个不同的索引i和j，使得nums[i]=nums[j]，并且i和j的差的绝对值至多为 k
仍然为滑动窗口类问题，用hashset辅助减少滑动窗口内元素查找的复杂度
*/
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        if(nums.size() == 0||k == 0) return false;
        int start = 0, end = 0, n = nums.size();
        unordered_set<int> s;
        while(end < n){
            if(s.count(nums[end])) return true;
            else s.emplace(nums[end]);
            if(end - start == k){
                s.erase(nums[start]);
                start++;
            }
            end++;
        }
        return false;
    }
};

/*220. 存在重复元素 III 在整数数组nums中，是否存在两个下标i和j，使得nums[i]和nums[j]的差的绝对值小于等于t，且满足i和j的差的绝对值也小于等于ķ。
如果存在则返回true，不存在返回false。
和上题类似的思路，难点在于使得nums[i]+nums[j]<=t，该条件可以转化为nums[i]-t<=nums[j]<=nums[i]+t；
借助set的lower_bound可以找到第一个大于等于nums[i]-t的数，则这个数再满足小于等于nums[i]+t即可
*/

class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        int start = 0, end = 0, n = nums.size();
        std::set<long> s;
        while(end < n){
            auto pos = s.lower_bound(long(nums[end]) - t);
            if(pos != s.end() && *pos <= long(nums[end]) + t) return true;
            else s.emplace(nums[end]);
            if(end - start == k){
                s.erase(nums[start]);
                start++;
            }
            end++;
        }
        return false;
    }
};