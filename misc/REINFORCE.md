```py
env = gym.make('CartPole-v0')

nn = torch.nn.Sequential(
    torch.nn.Linear(4, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, env_action_space.n),
    torch.nn.Softmax(dim=-1)
)

optim = torch.optim.Adam(nn.parameters(), lr=lr)


obs = torch.tensor(env.reset(), dtype=torch.float)
done = False
Actions, States, Rewards = [], [], []

While not done:
    probs = nn(obs)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample().item()
    obs_, rew, done = env.step(action)

    Actions.append(torch.tensor(action, dtype=torch.int))
    States.append(obs)
    Rewards.append(rew)

    obs = torch.tensor(obs_, dtype=torch.float)

DiscountedReturns = []
for t in range(len(Rewards)):
    G = 0.0
    for k,r in enumerate(Rewards[t:]):
        G += (gamma**k)*r # discount factor gamma is a hyperparameter
    DiscountedReturns.append(G)


for State, Action, G in zip(States, Actions, DiscountedReturns):
    probs = nn(State)
    dist = torch.distributions.Categorical(probs=probs)
    log_prob = dist.log_prob(Action)

    loss = -log_prob * G

    optim.zero_grad()
    loss.backward()
    optim.step()


obs = torch.tensor(env.reset(), dtype=torch.float)
done = False
env.render()

While not done:
    probs = nn(obs)
    c = torch.distributions.Categorical(probs=probs)
    action = c.sample().item()

    obs, rew, done, _info = env.step(action)
    env.render()

    obs = torch.tensor(obs_, dtype=torch.float)
```