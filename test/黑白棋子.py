'''
def f(n,m):
    if n==2 and m==2:
        return 12
    elif (n==1 and m==2) or (n==2 and m==1):
        return 2
    elif n==1 and m==1:
        return 0
    elif n==m:
        return 4*f(n-1,m-1)+2*n+2*m+4-4*f(n-2,m-1)+f(n-2,m-2)
    elif n>m:
        return 2*f(n-1,m)+2*m-f(n-2,m)
    elif m>n:
        return 2*f(n,m-1)+2*n-f(n,m-2)
a=int(input()) 
b=int(input())
print(f(a,b))
'''

###2 while测试
'''
def while_loop(n:int) ->  int:
    res=0
    i=1
    while i<=n:
        res +=i
        print("这是第{}个数".format(i))
        print(res)
        i+=1
    return res
m=int(input(""))
while_loop(m)
'''
###24年第一题测试
'''
def solveNQueens(n:int, m:int) -> int:
    def f(m,n):
        return m*n*(n-1)
    def g(m,n):
        return f(m-n+1,n)+2*n*(n-1)*(n-2)/3
    #print(int(f(m,n)+f(n,m)+g(m,n)*2))
    return f(m,n)+f(n,m)+g(m,n)*2
#测试part
#a=[]
#a=input("").split(" ")
#h=int(a[0])
#g=int(a[1])
#solveNQueens(h,g)
'''
###二分法查找#已默认排序完，不然要先sort一下
'''
def binary_search(li, val):
    left=0
    right=len(li)-1
    while left <=right:
        mid=(left+right)//2#此处为整除
        if li[mid]==val:
            print(mid)
            return mid
        elif li[mid]>val:
            right=mid-1
        elif li[mid]<val:
            left=mid+1
    else:
        print("None")
        return None
a=input("").split(" ")
b=input("")
binary_search(a,b)
'''
def num_integ(List[int])
