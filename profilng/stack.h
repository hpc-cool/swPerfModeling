#define	SIZE	10000000
unsigned long long tmp;
/*inline void init_stack(struct stack* s);
inline void push(struct stack* s, unsigned long long data);
inline unsigned long long pop(struct stack* s);*/
struct stack{
	unsigned long long sta[SIZE];
	int top;
};
//栈的初始化
inline void init_stack(struct stack* s)
{
	memset(s->sta, 0, sizeof(s->sta));
	s->top = 0;
	s->sta[s->top]=-1;
}
//压栈
inline void push(struct stack* s, unsigned long long data)
{
	/*if(s->top == SIZE-1){
		printf("the stack is full. \n");
		return;
	}	*/
 
	(s->top)++;
	s->sta[s->top] = data;
	return;

}
//出栈
inline unsigned long long pop(struct stack* s)
{
	if(s->top < 0 ){
		printf("the stack is empty. \n");
		return -1;
	} 
	else 
	{
		tmp = (s->sta)[s->top]; 
		(s->top)--;
		return  tmp;	
	}

}

inline int top(struct stack* s)
{
	return s->sta[s->top];
}
//栈的删除
void destory(struct stack *s)
{
	s->top = -1;
	memset(s->sta, 0, sizeof(s->sta));
}