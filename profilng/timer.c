#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <time.h>
#include "stack.h"
#include "rbtree.h"
#include <athread.h>
#include <dlfcn.h>
#include "mpi.h"

int __profile__funcID;//记录当前函数的编号

int __profile__fatherID=-1;//记录当前函数的父函数的编号

struct stack __profile__function;//存有从Main函数到当前函数所有调用函数编号的栈

struct stack __profile__time;//记录开始计时函数起始时间的栈

int __profile__rank;//记录当前处在哪个rank中（多进程）

int __profile_common_size;

struct node *T[9000];//函数运行信息记录节点

extern void init_stack(struct stack* s);
extern void push(struct stack* s, unsigned long long data);
extern unsigned long long pop(struct stack* s);

unsigned long addr2int[9001]; //记录函数地址的哈希值

//int_to_string 转换
char * myitoa (int n, char *s)
{
	int i,j,sign,len;
	len=0;
	if((sign=n)<0)//记录符号
		n=-n;//使n成为正数
	i=0;
	do{
		s[i++]=n%10+'0';//取下一个数字
		len++;
	}while((n/=10)>0);//删除该数字
	if(sign<0)
		s[i++]='-';
	for(j=0;j<len/2;j++)
	{
		char temp;
		temp=s[j];
		s[j]=s[len-j-1];
		s[len-j-1]=temp;
	}
	s[i]='\0';
	return s;
}


unsigned long long eax, edx,shelltimebegin,shelltime;
unsigned long long ic,oc;
unsigned long long __profile__record_time_begin()
{
	//记录开始计时函数的开始运行时间
	ic=athread_time_cycle();
	
	//栈初始化
	if(__profile__time.sta[__profile__time.top]!=-1&&__profile__time.top==0){
		init_stack(&__profile__time);
		int i;
		for(i=0;i<9000;i++)
		{
			T[i]=NULL;
		}
	}
	if(__profile__function.sta[__profile__function.top]!=-1&&__profile__function.top==0){
		init_stack(&__profile__function);
	}
	if(__profile__function.top>0)__profile__fatherID=top(&__profile__function);
	push(&__profile__function,__profile__funcID);
	push(&__profile__time,ic);

	//记录开始计时函数的结束运行时间
	oc=athread_time_cycle();
	push(&__profile__time,oc);
}

unsigned long long func_begin_time;

void __profile__record_time_end()
{
	
	unsigned long b = pop(&__profile__time);
	
	if(__profile__function.top>0) pop(&__profile__function);
	if(__profile__function.top>0) __profile__fatherID=top(&__profile__function);

	int search_result=search(T[__profile__funcID],__profile__fatherID,0,b);
	if(search_result==0)//若未被记录，则新建一个node，并插入红黑树
	{
		struct fatherNode __profile__node;
		__profile__node.fatherID=__profile__fatherID;
		__profile__node.times=1;
		ic=athread_time_cycle();
		__profile__node.accTime=ic - b;

		func_begin_time=pop(&__profile__time);
		oc=athread_time_cycle();
		__profile__node.shelltime=oc-func_begin_time;
		insert(&T[__profile__funcID],__profile__node);
	}

	if(__profile__time.top==0){
		MPI_Comm_size(MPI_COMM_WORLD, &__profile_common_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &__profile__rank);
		// printf("STACK is EMPTY!!%d\n",__profile_common_size);
		__profile__input_csv();
	}
}

//将函数运行信息写入硬盘
void __profile__input_csv()
{
	if(__profile__rank + 1!= __profile_common_size) return;
	char buf[1024];  
	FILE *fw;     
	char filename[1024] = "./out/", number[10];
	
	myitoa(__profile__rank+1, number);
	strcat(filename,number);
	strcat(filename,"_time.csv");
	// printf("%s\n",filename);
	fw = fopen(filename, "w");
	int i,j;
	for(i=1;i<9000+1;i++)
	{
		if(T[i]==NULL)continue;
		inorder(T[i],i,fw);
	}
    fclose(fw);
}

void __attribute__((no_instrument_function)) debug_log(const char *format,...);
void __attribute__((no_instrument_function)) __cyg_profile_func_enter(void*, void*);
void __attribute__((no_instrument_function)) __cyg_profile_func_exit(void*, void*);

#define MOD (unsigned long)(9000)
unsigned long addr2int[9001];

int getHashId(unsigned long oriId){
	int tmp=oriId%MOD;
	// printf("oriId:%ld\n",oriId);
	if(tmp==0)tmp++;
	while(addr2int[tmp]!=oriId){
		if(addr2int[tmp]==0||addr2int[tmp]==oriId){
			addr2int[tmp]=oriId;
			return tmp;
		}	
		tmp++;
		if(tmp>9000)tmp=1;
	}
	return tmp;
}
void  __attribute__((no_instrument_function))
__cyg_profile_func_enter(void *this, void *call)
{
	// printf("#yes enter:%ld %ld\n",call,this);
	int funId=getHashId((unsigned long)this), faFuncId=getHashId((unsigned long)call);
	// printf("#yes enter:%d %d\n",funId,faFuncId);
	__profile__funcID=funId;
	__profile__fatherID=faFuncId;
	__profile__record_time_begin();
    
}
 
void  __attribute__((no_instrument_function))
__cyg_profile_func_exit(void *this, void *call)
{
	// printf("#yes exit:%ld %ld\n",call,this);
	int funId=getHashId((unsigned long)this), faFuncId=getHashId((unsigned long)call);
	// printf("#yes exit:%d %d\n",funId,faFuncId);
	__profile__funcID=funId;
	__profile__fatherID=faFuncId;
	__profile__record_time_end();
}