# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # we need to iterate through the iterations to make sure that we go enough times
        for iteration in range(self.iterations):

            # creating a tempory counter to evaluate the states and put the values into.
            tempCounter = util.Counter()

            for state in self.mdp.getStates():  # iterating through the possible states

                maxValue = -9999999999999999999  # setting it to negative infinity

                for action in self.mdp.getPossibleActions(state):
                    probabilitiesOfTransStates = self.mdp.getTransitionStatesAndProbs(
                        state, action)

                    sigma = 0.0  # this will be the summation just like in the formula from class

                    # now we cycle through all of the probable states again just like the formula
                    # probablestate[1] = probability of next state
                    # probablestate[0] = next state
                    for probableState in probabilitiesOfTransStates:
                        sigma += probableState[1] * \
                            (self.mdp.getReward(
                                state, action, probableState[0]) + self.discount*(self.getValue(probableState[0])))
                    maxValue = max(maxValue, sigma)
                # checking to make sure that the max value has changed
                if maxValue != -9999999999999999999:
                    tempCounter[state] = maxValue
            self.values = tempCounter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for probableState in self.mdp.getTransitionStatesAndProbs(
                state, action):
            qValue += probableState[1] * \
                (self.mdp.getReward(
                    state, action, probableState[0]) + self.discount*(self.getValue(probableState[0])))

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        else:
            bestAction = None
            bestValue = -99999999  # we set this to negative infinity
            # we cycle through the actions to see which one provides the best value
            for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                if qValue > bestValue:
                    bestValue = qValue
                    bestAction = action
            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        counter = 0
        for iteration in range(self.iterations):
            # we only need to get one state and so we can pull that from our list of states
            # also if we hit the end we cycle back through the states we have already hit
            if(counter == len(self.mdp.getStates())):
                counter = 0
            state = self.mdp.getStates()[counter]
            maxValue = -999999
            if not self.mdp.isTerminal(state):
                # now we iterate over the actions and update only the one state
                for action in self.mdp.getPossibleActions(state):
                    maxValue = max(self.getQValue(state, action), maxValue)
                self.values[state] = maxValue
            counter += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # I googled it and a set is denoted with {}
        preds = {}

        # Step 1: finding all of the predecessors of all states
        for state in self.mdp.getStates():

            # checking to see if it is terminal
            if not self.mdp.isTerminal(state):

                # Now we need to go through possible actions
                for action in self.mdp.getPossibleActions(state):
                    for (newState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                        # we do not want duplicates so we check to see if the state is in there
                        if newState in preds:
                            preds[newState].add(state)
                        else:
                            preds[newState] = {state}

        # Step 2. Initialize a priority queue
        priorityQ = util.PriorityQueue()

        # Step 3. for each nonterminal state find the abs value of the difference and push it in negative form
        for state in self.mdp.getStates():
            # checking to see if it is a terminal state
            if not self.mdp.isTerminal(state):
                maxValue = -999999
                # iterate through the list of possible actions getting the max value
                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    maxValue = max(value, maxValue)
                diff = abs(self.values[state] - maxValue)

                # pushing to queue in negative form
                priorityQ.update(state, -diff)

        # Step 4. for iteration in iterations (follow documentation on UC berkley website)
        for iteration in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if priorityQ.isEmpty():
                break

            # Pop a state s off the priority queue.
            s = priorityQ.pop()

            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                maxVal = -99999999
                for action in self.mdp.getPossibleActions(s):
                    value = self.computeQValueFromValues(s, action)
                    maxVal = max(value, maxVal)
                self.values[s] = maxVal

            # For each predecessor p of s, do:
            for p in preds[s]:

                # Find the absolute value of the difference between the current value of p in self.values
                # and the highest Q-value across all possible actions from p
                if not self.mdp.isTerminal(p):
                    maxVal = -99999999
                    for action in self.mdp.getPossibleActions(p):
                        value = self.computeQValueFromValues(p, action)
                        maxVal = max(value, maxVal)
                    diff2 = abs(maxVal - self.values[p])

                    if diff2 > self.theta:
                        priorityQ.update(p, -diff2)
