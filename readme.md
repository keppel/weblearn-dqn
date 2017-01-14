<h1 align="center">
  <br>
  <a href="https://github.com/keppel/weblearn-dqn"><img src="https://cloud.githubusercontent.com/assets/1269291/21950583/6d22659c-d9b1-11e6-8fb4-2d61b196b688.gif" alt="WebLearn DQN" width="400" border="1"></a>
  <br>
  WebLearn DQN
  <br>
  <br>
</h1>

<h4 align="center">Simple Deep Q-learning agent for WebLearn.</h4>

<p align="center">
  <a href="https://www.npmjs.com/package/weblearn-dqn">
    <img src="https://img.shields.io/npm/dm/weblearn-dqn.svg"
         alt="NPM Downloads">
  </a>
  <a href="https://www.npmjs.com/package/weblearn-dqn">
    <img src="https://img.shields.io/npm/v/weblearn-dqn.svg"
         alt="NPM Version">
  </a>
</p>
<br>

Reinforcement learning agent that uses a [WebLearn] model to approximate the [Q-function] for your environment.

Q-learning is an **off-policy** algorithm, which means it can learn about the environment using trajectories where the actions weren't sampled from the agent (i.e. human demonstrator). I'll probably add a demo of this soon.

Q-learning is also a **model-free** algorithm, which means it's not doing any planning or tree search. It's basically just estimating the discounted future rewards it expects to see if takes an action **a** in state **s**, then taking the action with the highest expected value (or, sometimes, taking an action at random to explore some new part of the state space).

This implementation uses experience replay and temporal difference error clamping, but currently does **not** do fitted Q iteration ("target" network) or double DQN.

There's a demo using [OpenAI's gym] in `examples/`

## Usage

```
npm install weblearn weblearn-dqn
```

```js
const ndarray = require('ndarray')
const DQN = require('weblearn-dqn')
const { ReLU, Linear, MSE, SGD, Sequential } = require('weblearn')

let model = Sequential({
  optimizer: SGD(.01),
  loss: MSE()
})

const STATE_SIZE = 2
const NUM_ACTIONS = 3
// model input should match state size
// and have one output for each action
model.add(Linear(STATE_SIZE, 20))
     .add(ReLU())
     .add(Linear(20, NUM_ACTIONS))

let agent = DQN({
  model: model, // weblearn model. required.
  numActions: NUM_ACTIONS, // number of actions. required.
  epsilon: 1, // initial probability of selecting action at random (for exploration). optional.
  memorySize: 10000, // how many of our most experiences to remember for learning. optional.
  maxError: 1, // optionally limit the absolute value of the td-error from a single experience. false for no limit. optional.
  finalEpsilon: .1, // probability of selecting an action at random after `epsilonDecaySteps` steps of training. optional.
  epsilonDecaySteps: 100000, // on what timestep should we reach `epsilon === finalEpsilon`? optional.
  learnBatchSize: 32, // how many transitions should we learn from when we call agent.learn()? optional.
  gamma: .99 // factor used for discounting rewards far in the future vs. rewards sooner. optional.
})

// get these from your environment:
let observation = ndarray([.2, .74])
let reward = .3
let done = false

let action = agent.step(observation, reward, done)
// `action` is an integer in the range of [0, NUM_ACTIONS)

// call this whenever ya wanna do a learn step.
// you can call this after each `agent.step()`, but you can also call it more or less often.
// just keep in mind, depending on the size of your model, this may block for a relatively long time.
agent.learn()

```

[WebLearn]: https://github.com/keppel/weblearn
[Q-function]: https://en.wikipedia.org/wiki/Q-learning
[OpenAI's gym]: https://github.com/openai/gym
