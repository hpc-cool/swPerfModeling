#include <stdio.h>
#include <mpi.h>
void sayhello(){
    
    printf("hello world\n");

}
int main(){
    int comm_sz,my_rank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    int num=10;
    while(num--){
        sayhello();
    }
    MPI_Finalize();
    return 0;
}