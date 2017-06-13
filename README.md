# FoxNet

We explore a number of models and learning strategies for autonomous aircraft navigation and combat through the simplified environment of the Nintendo video game, Star Fox 64. Our primary model builds off DeepMind's 2015 DQN model. We first compare the performance of this model and three other neural architectures in an on-policy classification task, achieving up to 71\% validation accuracy with the DeepMind DQN model. We then improve these models with off-policy Deep Q-Learning. After 80,000 training iterations on Level 1, our best agent achieves an end-of-game score of 61 points, surpassing a random baseline's score of 12 but falling short of a human player's score of 115. We also demonstrate our agents' abilities to generalize game strategy to unseen levels. After 140,000 training iterations in ``Train Mode'', our best agent achieves an end-of-game score of 63 after only 10,000 additional training iterations on Level 1.

For more details, please see our paper (foxnet_final_paper.pdf).


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
