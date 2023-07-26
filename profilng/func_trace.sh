if [ $# != 3 ]; then
    echo 'Usage: addr2line.sh executefile addressfile functionfile'
    exit
fi
 
EXECUTABLE="$1"
TRACELOG="$2"
OUTLOG="$3"

cat $2 | while read myfunc
		do
            read fafunc
            echo -n function: >> $3
            echo -n `swaddr2line -e $1 -f $myfunc -s | sed -n '1p'` >> $3
            echo -n , called by function: >> $3
            echo -n `swaddr2line -e $1 -f $fafunc -s | sed -n '1p'` >> $3
            read timeline 
            echo $timeline >> $3 
		done
