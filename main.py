import gym
import time
import typing
import random
import gym
import torch
import torch.nn as nn
import copy


generation_quantity = 1000
population_size = 100
mutation_rate = 0.5
selection_percentage = 0.5
test_quantity = 5

env = gym.make("CartPole-v0")


def gen_random_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, 24),
        nn.Tanh(),
        nn.Linear(24, 48),
        nn.Tanh(),
        nn.Linear(48, 2),
        nn.Identity(),
    )


def mutate_net(net: nn.Sequential, mut: float) -> nn.Sequential:
    new_net = copy.deepcopy(net)
    for param in new_net.parameters():
        param.data += mut * torch.randn_like(param.data)
    return new_net


def multiple_test_nets(net: nn.Sequential) -> float:
    points = []
    for _ in range(test_quantity):
        points.append(test_net(net))
    return sum(points) / test_quantity


def test_net(net: nn.Sequential, image: bool = False) -> int:
    state = env.reset()

    done = False
    points = 0

    while not done:

        t_state = torch.tensor(state, dtype=torch.float)
        t_actions: torch.Tensor = net(t_state)
        action = int(torch.argmax(t_actions))

        new_state, _, done, _ = env.step(action)
        if image:
            env.render()
            time.sleep(0.05)
        points += 1

        state = new_state

    return points


def get_best(
    pop: typing.Dict[int, nn.Sequential], percentage: float
) -> typing.Tuple[int, typing.Dict[int, nn.Sequential]]:

    choosen_size = int(len(pop) * percentage)

    points = [(id, multiple_test_nets(net)) for id, net in pop.items()]
    ordered_points = sorted(points, key=lambda p: p[1], reverse=True)
    best_net_id = ordered_points[0][0]
    choosen_pop = {p[0]: pop[p[0]] for p in ordered_points[:choosen_size]}

    return best_net_id, choosen_pop


population = {id: gen_random_net() for id in range(population_size)}
next_id = population_size

for generation in range(generation_quantity):
    print(f"current generation: {generation}")
    print(f"population: {list(population.keys())}")
    best_net_id, best_nets = get_best(population, selection_percentage)
    print(f"best net id: {best_net_id}")
    best_net = population[best_net_id]
    points = test_net(best_net, image=True)
    print(f"points: {points}")
    while len(best_nets) < population_size:
        random_net = random.choice(list(best_nets.values()))
        new_net = mutate_net(random_net, mutation_rate)
        best_nets[next_id] = new_net
        next_id += 1
    population = best_nets
