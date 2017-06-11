# FoxNet

We explore a number of models and learning strategies for autonomous aircraft navigation and combat through the simplified environment of the Nintendo Star Fox 64 video game. Our primary model builds off DeepMind's 2015 DQN model. We first compare the performance of this model and three other neural architectures through an on-policy classification task, achieving up to 67\% validation accuracy with the DeepMind DQN model. We then improve these models with off-policy Deep Q-Learning, achieving end-of-game scores of up to 18 points.

A possible application of our system could be the control systems for an autonomous drone. In particular, training a drone agent in a simulated environment using the methods described in this paper could serve as an effective warm-start before training autonomous vehicles in the real world.


## Code

Note that our code lies on two different branches. For classification, check out the "main" branch; for q-learning, check out the "prime" branch.

The code in src/main/replay_buffer.py was borrowed from Assignment 2 in CS234 and src/main/saliency.py from Assignment 3 in CS231N. 

From this directory, run:

python src/main/run.py

Other optional arguments include:
--[dev,test]
--num_images ##
--eval_proportion ##
--save_data_dir "datadir/name"
(check out ./src/main/run.py for more)

For example:

/usr/bin/python2.7 src/main/run.py
--qlearning=True
--train_online=True
--user_overwrite=True
--verbose=True
--model=dqn


## Emulator

To run the emulator:
> cd src/emulator
> ./start.sh

To restart the emulator:
> netstat -nt
> sudo fuser -k 11111/tcp
> sudo fuser -k *****/tcp, where ***** is a number corresponding to 11111
[wait 30 seconds]
> ./start.sh
