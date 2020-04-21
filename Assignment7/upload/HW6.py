class Solution:
    def __init__(self):
        self.stones = None
    def lastStoneWeight(self, stones) -> int:
        stones = sorted(stones)
        self.stones = stones
        while self.stones is not None and len(self.stones) > 1:
            self.smash(self.stones)
        if self.stones is None:
            return 0
        else:
            return self.stones[0]
    def smash(self, stones):
        a, b = stones[-2:]
        if a == b:
            self.stones = stones[:-2]
        else:
            self.stones = stones[:-2] + [b - a]


if __name__ == '__main__':
    # stones = [2,7,4,1,8,1]
    # s = Solution()
    # a = s.lastStoneWeight(stones)
    # print(a)
    # s = "sadfgrweq"
    # print(s[0])
    s = "abcdefg"
    shift = [[1,1],[1,1],[0,2],[1,3]]
    s = list(s)
    n = len(s)
    for u in shift:
        d, a = u
        s += s
        if d == 0:
            a %= n
            s = s[a:a+n]
        else:
            a %= n
            a = n - a
            s = s[a: a + n]
    res = ""
    for s in s:
        res += s
    print(res)

    # s[-1:] = s
    # print(s)
