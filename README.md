# A3C (Under development)
A clean and minimal implementation of A3C (Asynchronous Advantage Actor Critic) algorithm in Pytorch.

## Versions

Discrete action space
* Single threaded version : ac_discrete.py
* Multithreaded synchronous version i. e. A2C : a2c_discrete.py
* Multithreaded asynchronous version i. e. A3C : a3c_discrete.py

Continuous action space
* Single threaded version : ac_continuous.py

## Tested on
* [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) (Discrete version from OpenAI Gym) - Move back and forth to balance a pole on a cart.

<p align="center">
<img src="media/a3c_cartpole.png" width="40%"/>
<img src="media/a3c_cartpole.gif" width="44%"/>
</p>

## References
* Mnih, Volodymyr, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. "Asynchronous methods for deep reinforcement learning." In International conference on machine learning, pp. 1928-1937. PMLR, 2016. [Link](https://arxiv.org/abs/1602.01783).
