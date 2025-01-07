INDX=2
STRIDE=8
PERCENTAGE=0.01

if [ $STRIDE -ne 0 ]
then
    for i in $(seq 0 $((STRIDE-1)))
    do
        screen -mS $((INDX+i)) sh screen_run.sh $((INDX+i)) $STRIDE $PERCENTAGE
        screen -d $((INDX+i))
    done
else
    screen -mS $INDX sh screen_run.sh $INDX $STRIDE $PERCENTAGE
    screen -d $INDX
fi