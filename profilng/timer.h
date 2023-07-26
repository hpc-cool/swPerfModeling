#ifndef TIMER_H_
#define TIMER_H_

#include "timer.h"
#include "stack.h"

#define MAX_MODULE 16

extern int __profile__funcID;

extern int __profile__fatherID;

extern struct stack __profile__function; //record funcID

extern struct stack __profile__time; //record runtime

extern int __profile__rank;


struct fatherNode
{
	unsigned int fatherID;
	unsigned long long accTime;  //根据下标确定module
	unsigned long long times;   //根据下标确定module
	unsigned long long shelltime;    //计时函数时间
};

#endif