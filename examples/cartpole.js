const Env = require('./lib/env.js')
const DQN = require('../')
const { ReLU, Linear, Sequential, SGD, MSE } = require('weblearn')

let env = Env('CartPole-v0')

let totalReward = 0
let numEpisodes = 0

env.on('ready', ()=> {
  let numActions = env.actionSpace.n
  let stateSize = env.observationSpace.shape[0]

  let model = Sequential({
    loss: MSE(),
    optimizer: SGD(.001)
  })

  model
  .add(Linear(stateSize, 40, false))
  .add(ReLU())
  .add(Linear(40, 40))
  .add(ReLU())
  .add(Linear(40, numActions, false))

  let agent = DQN({
    model,
    numActions,
    finalEpsilon: .1,
    epsilonDecaySteps: 10000,
    gamma: .9
  })

  env.on('observation', ({observation, reward, done, info})=> {
    let action = agent.step(observation, reward, done)
    if(reward)totalReward += reward
    if(done){
      numEpisodes++
      env.reset()
      console.log(totalReward / numEpisodes)
    } else {
      env.step(action, { render: false })
    }
  })
  env.reset()

  function learn() {
    setImmediate(()=> {
      agent.learn()
      learn()
    })
  }
  learn()

})

