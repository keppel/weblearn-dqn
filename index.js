const ndarray = require('ndarray')
const old = require('old')

function copy (target) {
  return ndarray(target.data.slice(), target.shape)
}

function indexOfMax (arr){
  if (arr.length === 0) {
    return -1
  }

  let max = arr[0]
  let maxIndex = 0

  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i
      max = arr[i]
    }
  }

  return maxIndex
}

function argmax (nd) {
  return indexOfMax(nd.data)
} 

function getRandomSubarray(arr, size) {
  size = Math.min(size, arr.length)
  let shuffled = arr.slice(0), i = arr.length, min = i - size, temp, index
  while (i-- > min) {
    index = Math.floor((i + 1) * Math.random())
    temp = shuffled[index]
    shuffled[index] = shuffled[i]
    shuffled[i] = temp
  }
  return shuffled.slice(min)
}

class DQN {
  constructor({
    model,
    numActions,
    epsilon,
    memorySize,
    maxError,
    evaluating,
    epsilonDecaySteps,
    finalEpsilon,
    learnBatchSize,
    gamma
  }) {
    this.numActions = numActions
    this.transitions = []
    this.model = model
    this.epsilon = typeof epsilon === 'number' ? epsilon : 1
    this.epsilonDecaySteps = epsilonDecaySteps || 100000
    this.finalEpsilon = typeof finalEpsilon === 'number' ? finalEpsilon : .2
    this.gamma = gamma || .99
    this.memorySize = memorySize || 10000
    this.maxError = maxError || false
    this.learnBatchSize = learnBatchSize || 32
    this.transitionCount = 0
  }

  step(currentState, currentReward, done) {
    let currentAction
    if(Math.random() < this.epsilon){
      currentAction = Math.floor(Math.random() * this.numActions)
    } else {
      const modelOutputs = this.model.forward(currentState)
      currentAction = argmax(modelOutputs)
    }

    if(this.previousState && typeof currentReward === 'number') {
      const transition = [
        this.previousState,
        this.previousAction,
        currentReward,
        currentState,
        done
      ]
      this.transitions[this.transitionCount % this.memorySize] = transition
      this.transitionCount++
    }

    this.previousState  = done ? null : currentState
    this.previousAction = done ? null : currentAction

    this.epsilon = Math.max(
      this.finalEpsilon, 
      this.epsilon - (1 / this.epsilonDecaySteps)
    )
    return currentAction
  }

  learn() {
    let transitions = getRandomSubarray(this.transitions, this.learnBatchSize)
    transitions.forEach((t, k) => {
      // q(s, a) -> r + gamma * max_a' q(s', a')
      const qPrime = copy(this.model.forward(t[3]))
      const q = this.model.forward(t[0])
      const target = copy(q)
      const reward = t[2]
      if(t[4]){
        target.data[t[1]] = reward
      } else {
        target.data[t[1]] = reward + this.gamma * qPrime.data[argmax(qPrime)]
      }

      const [loss, gradInputs] = this.model.criterion(q, target)

      if(this.maxError){
        gradInputs.data.forEach((v, k) => {
          if(Math.abs(v) > this.maxError){
            gradInputs.data[k] = v > 0 ? this.maxError : -this.maxError
          }
        })
      }

      this.model.backward(gradInputs)
      this.model.update()
    })
    return loss
  }
}

module.exports = old(DQN)