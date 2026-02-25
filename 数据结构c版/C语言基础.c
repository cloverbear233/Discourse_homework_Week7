#include<stdio.h>
/* 测试1
int main()
{
    int i=2,t=1;
    while(i<=5)
    {t=t*i;
    i+=1;}
    printf("%d\n",t);
    return 0;
}

/* 测试2
int main()
{
    int fenzi=1;
    double fenmu=2.0,zonghe=1.0,fenshu;
    while(fenmu<=100)
    {
        fenzi=-fenzi;
        fenshu=fenzi/fenmu;
        zonghe=fenshu+zonghe;
        fenmu+=1;

    }
    printf("%f\n",zonghe);
    return 0;
}
*/

//测试3
 
/* 函数外定义变量 x 和 y
int x;
int y;
int addtwonum()
{
    // 函数内声明变量 x 和 y 为外部变量
    extern int x;
    extern int y;
    // 给外部变量（全局变量）x 和 y 赋值
    x = 1;
    y = 2;
    return x+y;
}
 
//定义和print
//++	自增运算符，整数值增加 1	A++ 将得到 11
//--	自减运算符，整数值减少 1	A-- 将得到 9
#define PI 3.1415926//定义不可赋新值的常量
int piint=(int)PI;//将double转换为int类型
const float pi=3.1415926;//定义同样不可以赋新值更改的有名字常变量，关键字+数据类型+变量名=变量值
char c='?';//定义char类型变量
float a=3.14159f;//把此作为单精度浮点数定义，编译不出现警告
long double b=1.23L;//设置长双精度浮点数
int i=9,j;
int main(void){
    unsigned short price=50;
    printf("%u\n",price);
    printf("%d\n",piint);
    printf("%d %c\n",c,c);//一一对应，%d表示输出十进制整数（'a'的ASCII码为97），%c表示输出单个字符
    char c1='A';
    char c2=c1+32;//得到字符'a'的ASCII码
    printf("%c %d\n",c2,c2);//同样一一对应
    int result;
    result = addtwonum();// 调用函数 addtwonum
    printf("result\t为: %d ",result);//此处未换行,有\t为水平制表符
    printf("%f %Lf\n",a,b);
    printf("%d \n",(int)c1);//输出字符'A'的ASCII码
    j=++i;//先自增i，再赋值给j
    printf("%d %d\n",i,j);
    j=i++;//先赋值给j，再自增i
    printf("%d %d\n",i,j);
    return 0;
}
*/

 
/* 函数声明  全局变量
void func1(void); 
static int count=10;        //全局变量 - static 是默认的 
int main()
{
  while (count--) {//此处为循环，循环次数为count的值
      func1();//
  }
  return 0;
}
void func1(void)
{
// 'thingy' 是 'func1' 的局部变量 - 只初始化一次
  每次调用函数 'func1' 'thingy' 值不会被重置。
               
  static int thingy=5;
  thingy++;
  printf(" thingy 为 %d ， count 为 %d\n", thingy, count);
}
*/
/*位运算符
int main(){
 
   unsigned int a = 60;    // 60 = 0011 1100  
   unsigned int b = 13;    // 13 = 0000 1101 
   int c = 0;           
 
   c = a & b;       // 12 = 0000 1100  "与"运算
   printf("Line 1 - c 的值是 %d\n", c );
 
   c = a | b;       // 61 = 0011 1101 “或”运算 
   printf("Line 2 - c 的值是 %d\n", c );
 
   c = a ^ b;       // 49 = 0011 0001 “异或”运算
   printf("Line 3 - c 的值是 %d\n", c );
 
   c = ~a;          //-61 = 1100 0011  “取反”运算
   printf("Line 4 - c 的值是 %d\n", c );
 
   c = a << 2;     // 240 = 1111 0000 乘2的n次方
   printf("Line 5 - c 的值是 %d\n", c );
 
   c = a >> 2;     //15 = 0000 1111 除以2的n次方
   printf("Line 6 - c 的值是 %d\n", c );
   return 0;
}
*/
//##if 判断语句
//方法1：Exp1 ? Exp2 : Exp3;
/*
    int main(){
        int num;
        printf("输入一个数字：");
        scanf("%d",&num);//&表示将输入的内容赋值给num变量
        (num%2==0)?printf("偶数\n"):printf("奇数\n");
    }
*/
/*方法2：if(Exp1){Exp2;}else{Exp3;}
    int main(){
        int score;
        printf("请输入你的数据结构成绩：");
        scanf("%d",&score);
        if(score>=140){printf("超级优秀！\n");} //格式：if(){;}
        else if(score>=125&&score<140){printf("也很优秀！\n");}
        else {printf("再去练练吧！\n");}
        //套层if从句在{}里面再套if从句即可
    }
*/
/*方法3：switch case 语句
    int main(){
        int input;
        scanf("%d",&input);
        switch(input){
            case 1:
                printf("您输入的是1");
            case 2:
                printf("您输入的是2");
            default:
                printf("你输入不是1,2！");
        }
        return 0;
    }
*/
/*函数-while循环+函数-for循环
void print1_100(){ //void指函数没有返回值，括号里为空
        int i=1;
        while(1){ //此处也可以写i<=100，这样不用写break
            if(i==50){
                i++;
                continue;
            }
            else if(i==100){printf("%d\n",i);break;}
            else{
                printf("%d",i);
                i++;
            }
        }
    }
void print_1_100_for(){ //for 格式：for(初始化表达式;条件表达式;表达式){语句}
    for (int i=1;i<=100;i++){ //或把int i写在外面
        printf("%d",i); //for循环里面的break和continue用法与while一致
    int j=1;
    while (j<=10)printf("%d"",j++);
    }
}
void add1_100(){
    int i,j,sum=0;
    for(i=1,j=100;i<=50&&j>=51;i++,j--){
        sum=sum+i;
        sum=sum+j;
    }
    printf("\n%d\n",sum);
}
int main(){
    print1_100();
    print_1_100_for();
    add1_100();
    return 0;
}
*/

//指针
/*每一个变量都有一个内存位置，每一个内存位置都定义了可使用 & 运算符访问的地址，它表示了在内存中的一个地址。
int    *ip;    // 一个整型的指针 
double *dp;    //一个 double 型的指针 
float  *fp;    // 一个浮点型的指针 
char   *ch;    // 一个字符型的指针 
*/
/*!!指针也就是内存地址，指针变量是用来存放内存地址的变量。
int main ()
{
   int  var = 20;   // 实际变量的声明 
   int *ip; 
       // 指针变量的声明 其实这里*可以写成int* ip，ip是变量名（不包含*）
   ip = &var;  // 1写法 在指针变量ip中存储 var 的地址（ip变量存了var的地址）
   char ch='x';
   char ch1='y';
   char* q=&ch;//2 写法 这里的也可以写成char *q=&ch,q是变量名（不包含*），*是钥匙，*q对应ch（存储的变量的值）
   char *m=&ch1;//m是变量名（不包含*），*m对应ch（存储的变量的值）
   printf("ch1变量的地址:%p\n", &ch1);
   printf("ch变量的地址: %p\n", &ch);
   printf("var 变量的地址: %p\n", &var);
   // 在指针变量中存储的地址 var 变量的地址: 0x7ffeeef168d8
   printf("ip 变量存储的地址: %p\n", ip );//可以看出ip=&ch，&是取地址运算符
   // 使用指针访问值 ip 变量存储的地址: 0x7ffeeef168d8
   printf("*ip 变量的值: %d\n", *ip );
    // 使用指针访问值 *ip 变量的值: 20
   return 0;
}
*/

/*空指针
int main(){
    int* p=NULL;//指针没有指向任何变量，没有保存任何变量的地址
    if(p){printf("age=%d",*p);}//if(p)表示p不为空，if(!p)表示p为空
    else{printf("p is NULL,p的地址为%p\n",p);}
    int age=24;
    p=&age;
    if(p!=NULL){printf("age=%d\n",*p);}
    return 0;
}
*/

/*malloc函数和free函数
#include <stdlib.h>
int main(){
    int a=0;//静态分配内存，系统自动为变量a分配4个字节内存
    int *p=NULL;//定义整型指针p并初始化为NULL
    p=malloc(sizeof(int));//分配内存，返回指针，sizeof(int)用于计算括号内变量或数据型所占字节
    *p=10;//给指针指向的内存赋值
    printf("a=%d\n",a);//a=0
    printf("*p=%d\n",*p);// *p=10
    free(p);//释放内存，把指针p指向的内存空间释放了，而指针没有消失，存储的地址没有意义，p可以继续使用分配新的内存
    return 0;
}
*/


/*数组0
#define LENGTH(array)(sizeof(array)/sizeof(array[0]))//数组长度求法1
int main() {
    double balance[5] = {1000.0, 2.0, 3.4, 7.0, 50.0};
    //定义数组，元素个数为5，初始值为1000.0,2.0,3.4,7.0,50.0
    balance[4] = 50.0;//从0开始
    int n[3]={1,2,3};//定义数组，元素个数为3，初始值为1,2,3
    int m[3][2]={{1,2},{3,4},{5,6}};
    //定义二维数组，行数为3，列数为2，初始值为1,2,3,4,5,6
    int array[] = {1, 2, 3, 4, 5};//初始化数组，元素个数不确定，初始值为1,2,3,4,5
    int length = sizeof(array) / sizeof(array[0]);//数组长度求法2
    int length1=LENGTH(array);
    printf("数组长度为: %d\n", length);
    printf("数组长度为：%d\n",length1);
    return 0;
}
*/

/*数组2
#include <stdlib.h>
#define MAXSIZE 100
int main(){
    int A[10];//设置静态数组A，前提是知道数组元素个数
    int n;
    scanf("%d",&n);
    int* B=malloc(sizeof(int)*n);//动态数组B，不知道元素个数，
    char C[MAXSIZE];//定义静态数组C，元素个数为MAXSIZE
    for(int i=0;i<10;i++){ //初始化数组A
        A[i]=i;
        printf("%d ",A[i]);
    }
    printf("\n");
    for(int i=0;i<n;i++){
        B[i]=i+1;
        printf("%d ",B[i]);
    }
    printf("\n");
    free(B);
    return 0;
}
*/

/*字符（串）数组
#include<string.h>
#include <stdlib.h>
int main(){
    char A[4]={'h','a','h','a'};//这里只有A是字符数组，BC都是字符串数组，自带\0结尾，下面可以看到\0带来的影响
    char B[6]="23232";//这里的字符串长度为6，不包括结尾的'\0'，如果设置为[5]在输出的时候因为缺少\0会让B和A一起被输出
    char C[5];//定义字符数组C，元素个数为5，实际只能存放4个字符
    scanf("%s",C);//不建议超出4个，会导致数组（缓冲区）溢出，从下方地址存放来看，如果超出\0被占据会占据B的位置
    for(int i=0;i<4;i++){printf("%c",A[i]);}
    printf("\n");
    int m=sizeof(B)/sizeof(B[0]);
    printf("B的长度为：%d\n",m);
    for(int i=0;i<m;i++){printf("%c",B[i]);}
    printf("\n");
    int j=sizeof(C)/sizeof(C[0]);
    for (int i=0;i<j;i++){printf("%c",C[i]);}
    printf("\n");
    printf("A[3] address:%p\n",&A[3]);
    printf("B[0] address:%p\n",&B[0]);
    printf("B[4] address:%p\n",&B[4]);
    printf("C[0] address:%p\n",&C[0]);
    printf("String B is: %s\n",B);//字符串的占位符为%s
    printf("String C is: %s\n",C);
    return 0;//数组名就是数组的地址，在scanf输入的时候不用取地址符，
    //单个赋值需要写，例如scanf("%s",C);scanf("%c",&A[0]);
}
*/

/*二维数组输入
int main(){
    int A[2][3];
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            scanf("%d",&A[i][j]);
        }
    }
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            printf("%d",A[i][j]);
        }
    }
    printf("\n");
    return 0;
}
*/

/*数组3
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]); // 数组名arr被当作指针使用
    }
}

int main() {
    int myArray[5] = {10, 20, 30, 40, 50};
    printArray(myArray, 5); // 将数组名传递给函数
    return 0;
}
*/

/*结构体：可以组合多个数据结构变成一个整体
#include <stdlib.h>
struct student1{
        char name[20];
        int age;
        float weight;
        int score;
    }a;//可以定义为a的结构体，或者不写a，注意最后要有分号
    //给数据类型起个新名字
typedef struct student{
    char name[20];
    int age;
    float weight;
    int score;
}stu,*stup;//至此，struct student 就是stu的别名，stup是指针类型
//struct student a 变为stu a, struct student* b 变为 stup b

int main(){
    stu a;//定义结构体变量a
    stup b;//定义结构体指针变量b
    b=(stup)malloc(sizeof(stu));//创建b，将内存地址转换为stup类型
    scanf("%s",a.name);
    a.age=20;
    a.weight=65;//Tip1:结构体变量访问结构体内成员需要使用".""
    a.score=145;//用直接或者输入的方式给结构体a赋值
    printf("%s %d %f %d\n",a.name,a.age,a.weight,a.score);
    scanf("%s",b->name);//Tip2:结构体指针变量访问结构体内成员用"->"
    b->age=25;
    printf("%s的年龄为%d",b->name,b->age);
    free(b);
    return 0;
}

#define MAXSIZE 100
typedef struct{//这里省略了原结构体名
    int data[MAXSIZE];
    int length;
}SqList;
typedef struct LNode{//这里的链表结构体因为体内含有指针成员，原结构体名不能沈略，且要写原结构体不能写别名
    int data;
    struct LNode *next;
}LNode,*LinkList;//相当于完成typedef的重命名操作
*/

//函数
double area(double a, double b){//double为返回类型，area函数名，括号里形式参数
    double S;
    S=a*b; //函数体
    return S; //返回值
}
//函数名首位不能是数字，并且由字母/数字/下划线组成
/*
返回值可以有的特殊类型：
1 void 表示函数不返回任何值，直接写
void area(){
    printf();
    return;
}
2 bool 表示返回布尔值；
bool isEqual(int a, int b){
    return true/flase;
}
3 指针类型；
int* createArray(int size){
    int* arr=(int*)malloc(sizeof(int)*size);
    return arr/NULL;
}
*/
//查找的void print做法
void Search(int arr[],int n,int x){
    int m=0;
    for (int i=0;i<n;i++){
        if(arr[i]==x){
            printf("%d ",i);m=1;}}
    if (m==0){printf("-1\n");}
    return;
}
//查找的一般return 做法(目前只能返回出一个)
int Search1(int arr[],int n,int x){
    for (int i=0;i<n;i++){
        if(arr[i]==x){return i;}}
    return -1;
}
 int main(){
    int arr[5]={6,8,6,2,1};
    int x;
    scanf("%d",&x);
    Search(arr,5,x);
    int y=Search1(arr,5,x);
    printf("\n第二种方法测定结果：%d\n",y);
    return 0;
 }