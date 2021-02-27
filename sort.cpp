// 排序：
// 其余更一般的排序算法：https://www.runoob.com/w3cnote/ten-sorting-algorithm.html 
/*分桶排序*/
class Solution {
public:
  vector<int> sortArray(vector<int>& nums) {
    int min_val = 500001, max_val = -500001, idx = 0;
    for(auto x: nums) {
      min_val = min(min_val, x);
      max_val = max(max_val, x);
    }
    vector<int> bins(max_val - min_val + 1, 0);
    for(auto x: nums) {
      bins[x - min_val] ++;
    }
    for(int x = min_val; x <= max_val; x ++) {
      int c = bins[x - min_val];
      while(c -- > 0) {
        nums[idx ++] = x;
      }
    }
    return nums;
  }
};

/*归并排序*/
class Solution {
    vector<int> tmp;
    void mergeSort(vector<int>& nums, int l, int r) {
        if (l >= r) return;
        int mid = (l + r) >> 1;
        mergeSort(nums, l, mid);
        mergeSort(nums, mid + 1, r);
        int i = l, j = mid + 1;
        int cnt = 0;
        while (i <= mid && j <= r) {
            if (nums[i] < nums[j]) {
                tmp[cnt++] = nums[i++];
            }
            else {
                tmp[cnt++] = nums[j++];
            }
        }
        while (i <= mid) {
            tmp[cnt++] = nums[i++];
        }
        while (j <= r) {
            tmp[cnt++] = nums[j++];
        }
        for (int i = 0; i < r - l + 1; ++i) {
            nums[i + l] = tmp[i];
        }
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        tmp.resize((int)nums.size(), 0);
        mergeSort(nums, 0, (int)nums.size() - 1);
        return nums;
    }
};

/*快速排序1*/
class Solution {
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }
    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l; // 随机选一个作为我们的主元
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }
    void randomized_quicksort(vector<int>& nums, int l, int r) {
        if (l < r) {
            int pos = randomized_partition(nums, l, r);
            randomized_quicksort(nums, l, pos - 1);
            randomized_quicksort(nums, pos + 1, r);
        }
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        srand((unsigned)time(NULL));
        randomized_quicksort(nums, 0, (int)nums.size() - 1);
        return nums;
    }
};
/*快速排序2*/
class Solution {
  void quickSort(vector<int>& nums, int l, int r){
        if(l >= r) return;
        int pivot = nums[l], i = l, j = r;
        while(i < j){
            while(i < j && nums[j] >= pivot) j--;
            while(i < j && nums[i] <= pivot) i++;
            if(i < j) swap(nums[i], nums[j]);
        }
        nums[l] = nums[j];
        nums[j] = pivot; 
        quickSort(nums, l, j - 1);
        quickSort(nums, j + 1, r);
    }
public:
  vector<int> sortArray(vector<int>& nums) {
    quicksort(nums, 0, nums.size() - 1);
    return nums;
  }
};

/*堆排序*/
class Solution{
public:
void maxHeapify(vector<int>& nums, int start, int end){
    int parent = start, child = 2 * start + 1;
    while(child <= end){
      if(child + 1 <= end && nums[child] < nums[child + 1]) child++;
      if(nums[parent] < nums[child]){
          swap(nums[parent], nums[child]);
          parent = child;
      }
      child = 2 * child + 1;
    }
}

void heapSort(vector<int>& nums){
    int n = nums.size();
    for(int i = n/2 - 1;i >= 0;i--) maxHeapify(nums, i, n-1);
    for(int i = n - 1;i > 0;i--){
        swap(nums[0], nums[i]);
        maxHeapify(nums, 0, i - 1);
    } 
}
};

class Solution {
    
public:
    void maxHeapify(vector<int>& nums, int start, int end){
        int largest = start, l = 2 * start + 1, r = 2 * start + 2;
        if(l < end && nums[start] < nums[l]) largest = l;
        if(r < end && nums[largest] < nums[r]) largest = r;
        if(largest != start){
            swap(nums[start], nums[largest]);
            maxHeapify(nums, largest, end);
        }
    }
    void heapSort(vector<int>& nums){
        int n = nums.size();
        for(int i = n/2 - 1;i >= 0;i--) maxHeapify(nums, i, n-1);
        for(int i = n - 1;i > 0;i--){
            swap(nums[0], nums[i]);
            maxHeapify(nums, 0, i - 1);
        } 
    }
    vector<int> sortArray(vector<int>& nums) {
        heapSort(nums);
        return nums;
    }
};

// 分治
/*4. 寻找两个正序数组的中位数*/
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        if(m + n == 0) return 0;
        if(m > n) return findMedianSortedArrays(nums2, nums1);

        int imin = 0, imax = m;
        int maxLeft, maxRight;
        while(imin <= imax){
            int i = (imin + imax)/2;
            int j = (m + n + 1)/2 - i;
            if(i < m && nums2[j - 1] > nums1[i])
                imin = i + 1;
            else if(i > 0 && nums1[i - 1] > nums2[j])
                imax = i - 1;
            else{
                if(i == 0) maxLeft = nums2[j - 1];
                else if(j == 0) maxLeft = nums1[i - 1];
                else maxLeft = max(nums1[i - 1], nums2[j - 1]);
                if((m + n) % 2) return maxLeft;
                if(i == m) maxRight = nums2[j];
                else if(j == n) maxRight = nums1[i];
                else maxRight = min(nums1[i], nums2[j]);
                return 1.0*(maxLeft + maxRight)/2;
            }
        }
        return 0;
    }
};

/*找两个有序数组的第K小元素*/
class Solution {
public:
    int findKthElm(vector<int>& nums1,vector<int>& nums2,int k){
        assert(1<=k&&k<=nums1.size()+nums2.size());
        int le=max(0,int(k-nums2.size())),ri=min(k,int(nums1.size()));
        while(le<ri){
            int m=le+(ri-le)/2;
            if(nums2[k-m-1]>nums1[m]) le=m+1;
            else ri=m;
        }//循环结束时的位置le即为所求位置，第k小即为max(nums1[le-1]),nums2[k-le-1])，但是由于le可以为0、k,所以
        //le-1或者k-le-1可能不存在所以下面单独判断下
        int nums1LeftMax=le==0?INT_MIN:nums1[le-1];
        int nums2LeftMax=le==k?INT_MIN:nums2[k-le-1];
        return max(nums1LeftMax,nums2LeftMax);
    }
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n=nums1.size()+nums2.size();
        if(n&1){//两个数组长度和为奇数
            return findKthElm(nums1,nums2,(n>>1)+1);
        }
        else{//为偶数
            return (findKthElm(nums1,nums2,n>>1)+findKthElm(nums1,nums2,(n>>1)+1))/2.0;
        }
    }
};

/*
给定一个规则的list, 形如山峰（满足单峰，且峰值点有可能在端点）：[1,3,9,10,8,6,2]，可能存在重复值
请写出程序，返回第k大的数，
不使用内置排序程序
*/
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int idx = searchPeakIndex(nums);//二分查找找到峰值元素索引，该元素为最大元素
        int i = idx - 1, j = idx + 1, c = 1;
        vector<int> ans; ans.push_back(nums[idx]);
        while(i >= 0 && j <= nums.size() && c <= k) {//合并两个有序数组
          if(nums[i] > nums[j]) {
            ans.push_back(nums[i]);
            ans.push_back(nums[j]);
          } else {
            ans.push_back(nums[j]);
            ans.push_back(nums[i]);
          }
          c += 2;
        }
        return ans[k - 1];
    }

    int searchPeakIndex(vector<int>& nums) {
      int l = 0, r = nums.size() - 1;
      if(nums[l] > nums[l + 1])
        return l;
      else if(nums[r] > nums[r - 1])
        return r;
      while(l <= r) {
        int pivot = (l + r) >> 2;
        if(nums[pivot - 1] < nums[pivot] && nums[pivot] > nums[pivot + 1])
          return pivot;
        if(nums[pivot - 1] < nums[pivot] && nums[pivot] < nums[pivot + 1])
          l = pivot + 1;
        else if((nums[pivot - 1] > nums[pivot] && nums[pivot] > nums[pivot + 1]))
          r = pivot - 1;
      }
      return -1;
    }
};

/*440. 字典序的第K小数字
问题可以拆解成几个子问题：
1.一个前缀下有多少数字
2.第k小数子的前缀大于当前前缀，如何扩大前缀，增大寻找的范围？
3.第k小数子的前缀在当前前缀下，怎么继续往下面的子节点找？
*/
class Solution {
public:
    long getCount(long prefix, long n){
        long curr = prefix;
        long next = prefix + 1;
        long count = 0;
        while(curr <= n){
            count += min(n + 1, next) - curr;
            curr *= 10;
            next *= 10;
        }
        return count;
    }
    int findKthNumber(int n, int k) {
        long pos = 1, prefix = 1;
        while(pos < k){
            long count = getCount(prefix, n);
            if(pos + count > k){
                prefix *= 10;
                pos++;
            }else{
                prefix++;
                pos += count;
            }
        }
        return static_cast<int>(prefix);
    }
};


/*386. 字典序排数*/
class Solution {
public:
    void dfs(vector<int>& nums, int prefix, int n){
        if(prefix > n) return;
        nums.push_back(prefix);
        for(int i = 0;i < 10;i++){
            dfs(nums, prefix*10 + i, n);
        }
    }
    vector<int> lexicalOrder(int n) {
        vector<int> nums;
        for(int i = 1;i < 10;i++){
            dfs(nums, i, n);
        }
        return nums;
    }
};

/****牛课网：请实现有重复数字的有序数组的二分查找。*/
int binarySearch(vector<int>& nums, int target){
        int n = nums.size();
        int low = 0, high = n;
        while(low < high){
            int mid = (low + high) / 2;
            if(nums[mid] < target) low = mid + 1;
            if(nums[mid] >= target) high = mid;
        }
        return low;
}

/*34. 在排序数组中查找元素的第一个和最后一个位置*/
class Solution {
public:
    int binarySearch(vector<int>& nums, int target){
        int n = nums.size();
        int low = 0, high = n;
        while(low < high){
            int mid = (low + high) / 2;
            if(nums[mid] < target) low = mid + 1;
            if(nums[mid] >= target) high = mid;
        }
        return low;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        int n = nums.size();
        if(n == 0) return {-1, -1};
        if(n == 1 && nums[0] == target) return {0, 0};
        int start = binarySearch(nums, target);
        int end = binarySearch(nums, target + 1) - 1;
        if(start < n && nums[start] == target) 
            return {start, end};
        else return{-1, -1};
    }
};