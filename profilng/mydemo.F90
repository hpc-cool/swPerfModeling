!#include "../fortran/config.h"
module mytest
  use openarray
  !use iso_c_binding
  !use oa_mod

contains

  subroutine array_creation()
    use mpi
    use openarray
    implicit none
    integer :: ierr,i,p,q,r,j
    type(array) :: A, B, C, D,h,dy,dx,north_e,fsm,dum,dvm,art,aru,arv,tb
   
    real :: aa(3, 3), bb(3, 3), cc(3, 3)

    character(len=1000) :: fnc
    fnc="grid.nc"
    !A=seqs(4,5,6)
    !B=A
    !call display(B,"B=")

    !return;
    A = seqs(1,2,3)
    B = seqs(1,2,3)
    D = seqs(1,2,3)
    !call display(A,"A=")
    !call display(B,"B=")
    ! call display(D,"D=")
    print * ,"123"
    C = A*B
    !call display(C,"C=")
    print * ,"456"

    call tic("AplusB")  ! start the timer
    !do i=1,1000
    !C = A+B
    !enddo
    call toc("AplusB")  ! start the timer
  end subroutine

end module

program main
  use openarray
  use mpi
  use mytest
    
  implicit none
  integer :: step
  integer :: i, nt, nx, ny, nz
  ! initialize OpenArray, no split in z-direction
  print *,__FILE__,__LINE__
  call oa_init(MPI_COMM_WORLD, [-1, -1, 1])
do i = 1, 5 
  call array_creation()
end do
  print *,__FILE__,__LINE__
  !if(get_rank() .eq. 0)call show_timer()
  call oa_finalize()
end program main

