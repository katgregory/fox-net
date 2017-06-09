# FoxNet

To run, from this directory, run:

python src/main/run.py

Other optional arguments include:
--[dev,test]
--num_images ##
--eval_proportion ##
--data_dir "datadir/name"


For example:

/usr/bin/python2.7 src/main/run.py
--qlearning=True
--train_online=True
--user_overwrite=True
--verbose=True
--model=fc
--epsilon=0.05


To run the emulator:
> cd src/emulator
> ./start.sh



To restart the emulator:
> netstat -nt
> sudo fuser -k 11111/tcp
> sudo fuser -k *****/tcp, where ***** is a number corresponding to 11111
[wait 30 seconds]
> ./start.sh