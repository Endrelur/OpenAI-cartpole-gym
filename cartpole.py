import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm


# Create bins and Q table
def create_bins_and_q_table(env, numBins=30):

    obsSpaceSize = len(env.observation_space.high)

    # Get the size of each bucket
    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-0.418, 0.418, numBins),
        np.linspace(-4, 4, numBins),
    ]

    qTable = np.random.uniform(
        low=-2, high=0, size=([numBins] * obsSpaceSize + [env.action_space.n])
    )

    return bins, obsSpaceSize, qTable

# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(
            np.digitize(state[i], bins[i]) - 1
        )  # -1 will turn bin into index
    return tuple(stateIndex)

# Q-learn
def Q_learn(
    qTable,
    env,
    bins,
    obsSpaceSize,
    runs,
    learningRate=0.1,
    discount=0.95,
    showEvery=1000,
    updateEvery=100,
    printStats=True,
):
    """
    A function that applies QLearning to agiven enviroment with a given Q-Table.


    qTable -- The Q-table that is going to be trained in the form of a Numpy array.

    env -- The OpenAI-enviroment containing the action space that we can observe and act upon.

    bins -- The bins containing the buckets.

    obsSpaceSize -- The size of the observation space.

    runs -- The amount of runs to perform training.

    learningRate -- How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded (default = 0.1).

    discount -- Between 0 and 1, mesue of how much we carre about future reward over immedate reward (default = 0.95).

    showEvery -- How many runs between each visualization of the enviroment (default = 1000).

    updateEvery -- How many runs between each print to the console describing run,average score, min-score and max-score (default = 100).

    printStats -- Wether or not the stats is printed to the terminal at updateEvery-intervals (default = True).


    returns a metrics object with the following array attributes: 'ep' (episode number), 'avg' (average score at that point), 'min'(minimum score at that point), max (maximum score at that point).
    """

    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    startEpsilonDecay = 1
    endEpsilonDecay = runs // 2
    epsilon_decay_value = epsilon / (endEpsilonDecay - startEpsilonDecay)

    previousCnt = []  # array of all scores over runs
    metrics = {"ep": [], "avg": [], "min": [], "max": []}  # metrics recorded for graph

    for run in tqdm(range(runs)):
        discreteState = get_discrete_state(env.reset(), bins, obsSpaceSize)
        done = False  # has the enviroment finished?
        cnt = 0  # how may movements cart has made

        while not done:
            if run % showEvery == 0:
                env.render()  # if running RL comment this out

            cnt += 1
            # Get action from Q table
            if np.random.random() > epsilon:
                action = np.argmax(qTable[discreteState])
            # Get random action
            else:
                action = np.random.randint(0, env.action_space.n)
            newState, reward, done, _ = env.step(action)  # perform action on enviroment

            newDiscreteState = get_discrete_state(newState, bins, obsSpaceSize)

            maxFutureQ = np.max(
                qTable[newDiscreteState]
            )  # estimate of optiomal future value
            currentQ = qTable[discreteState + (action,)]  # old value

            # pole fell over / went out of bounds, negative reward
            if done and cnt < 200:
                reward = -375

            # formula to caculate all Q values
            newQ = (1 - learningRate) * currentQ + learningRate * (
                reward + discount * maxFutureQ
            )
            qTable[discreteState + (action,)] = newQ  # Update qTable with new Q value

            discreteState = newDiscreteState

        previousCnt.append(cnt)

        # Decaying is being done every run if run number is within decaying range
        if endEpsilonDecay >= run >= startEpsilonDecay:
            epsilon -= epsilon_decay_value

        # Add new metrics for graph
        if run % updateEvery == 0:
            latestRuns = previousCnt[-updateEvery:]
            averageCnt = sum(latestRuns) / len(latestRuns)
            metrics["ep"].append(run)
            metrics["avg"].append(averageCnt)
            metrics["min"].append(min(latestRuns))
            metrics["max"].append(max(latestRuns))
            if printStats:
                print(
                    "Run:",
                    run,
                    "Average:",
                    averageCnt,
                    "Min:",
                    min(latestRuns),
                    "Max:",
                    max(latestRuns),
                )
    env.close()
    return metrics, qTable


# Setting up and running the enviroment.
env = gym.make("CartPole-v0")
bins, obsSpaceSize, qTable = create_bins_and_q_table(env, numBins=20)
metrics, trained_qTable = Q_learn(
    qTable,
    env,
    bins,
    obsSpaceSize,
    11000,
    discount=0.995,
    printStats=False,
    learningRate=0.05,
)

# Plot graph using the metrics returned from training
plt.plot(metrics["ep"], metrics["avg"], label="average rewards")
plt.plot(metrics["ep"], metrics["min"], label="min rewards")
plt.plot(metrics["ep"], metrics["max"], label="max rewards")
plt.legend(loc=4)
plt.show()

env = gym.make("CartPole-v0")
bins,obsSpaceSize,newTable=create_bins_and_q_table(env, numBins=20)
discreteState = get_discrete_state(env.reset(), bins, obsSpaceSize)

for test in tqdm(range(2)):
    obs = env.reset()
    done = False
    rew = 0
    while done != True:
        action = np.argmax(trained_qTable[discreteState])
        newState, reward, done, _ = env.step(action)  # perform action on enviroment
        discreteState = get_discrete_state(newState, bins, obsSpaceSize)
        rew += reward
        sleep(0.03)
        env.render()
    print("episode : {}, reward : {}".format(test, rew))
