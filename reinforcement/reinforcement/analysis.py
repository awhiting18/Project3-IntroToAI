# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # I tried all of the values for this (i.e. 0-1 in steps of .1) and there was no postive effect
    answerDiscount = .9
    # I tried .1 and no effect. I made it 0 and it worked. I also tried .01 and that worked but .05 did not
    answerNoise = 0.01
    return answerDiscount, answerNoise


def question3a():
    answerDiscount = .5
    answerNoise = .1
    # my thought behind this is if we want the close reward we need to deduct heavily for living
    answerLivingReward = -3
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3b():
    # I kept the same thinking as the last one but made the answer discount less
    answerDiscount = .1
    answerNoise = .1
    answerLivingReward = -3
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3c():
    # I removed the noise and the test passed
    answerDiscount = .9
    answerNoise = 0
    answerLivingReward = -.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3d():
    answerDiscount = .1
    answerNoise = .2
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    # I made the living reward huge so that the agent would never want to leave
    answerDiscount = .1
    answerNoise = 0
    answerLivingReward = 20
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question8():
    answerEpsilon = 2
    answerLearningRate = 1
    return 'NOT POSSIBLE'
    # I tried multiple different variations but came to the conclusion that it wasn't possible
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
