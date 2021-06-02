#!/bin/bash

############## SLURM SBATCH OPTIONS (not using #SBATCH --exclusive)
#SBATCH -o <custom>.sh.log-%j
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

############## ARGUMENTS
ROOT_DIR=${HOME}/<custom>
CONFIG=${ROOT_DIR}/baselines/config/<custom>
LOCAL_DIR=${HOME}/<custom>
TEMP_DIR=${HOME}/<custom>
CHECKPOINT_FREQ=50
# specify RESTORE=<path-to-checkpoint> here to restore from a checkpoint

# use xvfb for x-server (CPU-only mode); this only support no-graphics mode for unity build
# as mesa-20.0 only support OpenGL 3.1.0 but Unity require 3.3+
# USE_XVFB=true 

############## LOAD MODULE
source /etc/profile
#module load anaconda/2020a # comment this out if you want to use your own anaconda
printf "\n"
printf "%s\n" "Modules Successfully Loaded"

############## SETUP X-SERVER
if ! $USE_XVFB
then
   startx & # start x server
   sleep 10s
   printf "\n"
   printf "%s\n" "Started X"

   DISPLAYNUM=$(ps -efww | grep xinit | grep -v grep | sed -re "s/^.* :([0-9]+) .*$/\1/g")
   if [ -z "${DISPLAYNUM}" ]; then
      printf "%s\n" "Could not find my assigned display number from process list!"
      exit 1
   elif [[ ! "$DISPLAYNUM" =~ ^[0-9]+$ ]]; then
      printf "%s\n" "Got non-numeric value for DISPLAYNUM: $DISPLAYNUM"
      exit 1
   elif [ ! -e /tmp/.X11-unix/X${DISPLAYNUM} ]; then
      printf "%s\n" "My DISPLAY is apparently $DISPLAYNUM, but there is no socket at /tmp/.X11-unix/X${DISPLAYNUM}!"
      exit 1
   fi
   export DISPLAY=:${DISPLAYNUM}

   printf "\n"
   printf "%s\n" "Exported Display :${DISPLAYNUM}"
   xset q # to check if x server works
fi

############## RUN SCRIPT 
source ${HOME}/.bashrc
source activate airguardian

printf "\n"
printf "Root Directory: %s\n" ${ROOT_DIR}
printf "Config: %s\n" ${CONFIG}
printf "Local Dirctory: %s\n" ${LOCAL_DIR}
printf "Temp Dirctory: %s\n" ${TEMP_DIR}
printf "Checkpoint Frequency: %s\n" ${CHECKPOINT_FREQ}

printf "\n"
printf "%s\n" "Start Script"
if ! $USE_XVFB
then
   if [ -z "$RESTORE" ]
   then
      python ${ROOT_DIR}/baselines/train.py -f ${CONFIG} --local-dir ${LOCAL_DIR} --temp-dir ${TEMP_DIR} \
                                          --checkpoint-freq ${CHECKPOINT_FREQ} --checkpoint-at-end
   else
      printf "Restore from %s\n" ${RESTORE} 
      python ${ROOT_DIR}/baselines/train.py -f ${CONFIG} --local-dir ${LOCAL_DIR} --temp-dir ${TEMP_DIR} \
                                          --checkpoint-freq ${CHECKPOINT_FREQ} --checkpoint-at-end \
                                          --restore ${RESTORE}
   fi
else
   if [ -z "$RESTORE" ]
   then
      xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python ${ROOT_DIR}/baselines/train.py \
                                          -f ${CONFIG} --local-dir ${LOCAL_DIR} --temp-dir ${TEMP_DIR} \
                                          --checkpoint-freq ${CHECKPOINT_FREQ} --checkpoint-at-end
   else
      printf "Restore from %s\n" ${RESTORE} 
      xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python ${ROOT_DIR}/baselines/train.py \
                                          -f ${CONFIG} --local-dir ${LOCAL_DIR} --temp-dir ${TEMP_DIR} \
                                          --checkpoint-freq ${CHECKPOINT_FREQ} --checkpoint-at-end \
                                          --restore ${RESTORE}
   fi
fi

############## COMPLETE AND CLOSE X-SERVER
printf "\n"
printf "%s\n" "Run Complete"
if ! $USE_XVFB
then
   killall startx
   killall xinit
   printf "%s\n" "Closed X Sessions"
fi