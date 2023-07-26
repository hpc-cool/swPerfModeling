oplevel2 = -O2 -w  -std=c++0x -D_WITHOUT_LLVM_ -DSUNWAY  -fpermissive -faddress_align=64

FC = mpif90 ${oplevel2} -mieee 
CC = mpicc ${oplevel2} -O2 -w -mieee

LIBS =  -lstdc++ -ldl

FCFLAGS = -Wno-tabs -g -finstrument-functions \
		  -fbacktrace -ffree-line-length-0 

mydemo: mydemo.o timer.o mpiwrapper.o mpiwrapper_f.o
	${CC} -o mydemo mydemo.o timer.o mpiwrapper.o mpiwrapper_f.o ${EXT_LIB_LINK} ${LIBS} -L. -lm -lm_slave -ldl -lstdc++

timer.o: timer.c
	-${CC} -c -o $@ $<

mydemo.o: mydemo.c
	$(CC) -g -finstrument-functions -c -o $@ $<

mpiwrapper.o: mpiwrapper.c
	-${CC} -g -finstrument-functions -c -o $@ $<

mpiwrapper_f.o: mpiwrapper_f.c
	-${CC} -g -finstrument-functions -c -o $@ $<


.PHONY:clean

clean:
	-rm mydemo.o
	-rm mydemo
	-rm timer.o

run :
	-rm ./out/2_time.csv
	-rm ./out/func.out
	bsub -I -q q_sw_expr -share_size 7000 -host_stack 4096 -n 2 -cgsp 64 ./mydemo
	sh func_trace.sh mydemo ./out/2_time.csv ./out/func.out