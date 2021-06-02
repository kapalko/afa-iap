### BUG doesn't correctly setup after running script on command line
#!/bin/bash
source /etc/profile 

startx & # start x server
sleep 5s
DISPLAYNUM=$(ps -efww | grep xinit | grep -v grep | sed -re "s/^.* :([0-9]+) .*$/\1/g")
export DISPLAY=:${DISPLAYNUM}

echo $DISPLAY # to check if x server works
xset q # to check if x server works

source ~/.bashrc
sleep 1s
conda activate airguardian
