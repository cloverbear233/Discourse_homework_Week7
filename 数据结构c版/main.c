#include <stdio.h>//stdio.h 是 C 语言标准库中的一个头文件，
/*代表“标准输入输出”（Standard Input Output）,这样能使用scanf,printf等函数*/
/*这里的写法近似于在python里面import库
*/
// 函数声明
int add(int a, int b);

int main() {
    int num1, num2, sum;
    int add(int a, int b); // 函数声明,不必要,但为了完整性,可以写上
    // 用户输入两个整数
    printf("请输入第一个整数: ");//输出提示信息
    scanf("%d", &num1);/*scanf是输入函数，第一个参数是格式字符串，
                    第二个参数是地址，存放输入值,&num1表示num1的地址*/
    printf("请输入第二个整数: ");
    scanf("%d", &num2);

    // 调用函数并计算和
    sum = add(num1, num2);

    // 输出结果
    printf("两个整数的和是: %d\n", sum);

    return 0;//用于主函数 main() 的结尾，表示程序的正常结束
}

// 函数定义
int add(int a, int b) {
    return a + b;
}
