"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I lowered the value of noise from 0.2 to 0.01. This causes the agent to be more determinsitic
    and less random.
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise

def question3a():
    """
    Lowered living reward so the agent could take the more risky path
    Lowered noise so the agent could be more deterministic
    Lowered discount so the agent would want immediate rewards (close exit)
    """

    answerDiscount = 0.3
    answerNoise = 0.01
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Lowered the discount factor so the agent would want the closer
    exit while increasing
    the living reward for the agent to take a safer path.
    """

    answerDiscount = 0.01
    answerNoise = 0.01
    answerLivingReward = 0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Increased discount so agent would take a longer path for a
    higher reward and
    increased living reward so agent would take a safer path.
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Set living reward back to 0 so the agent would take a safer path
    """

    answerDiscount = 0.9
    answerNoise = 0.3
    answerLivingReward = 0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Raised living reward so the agent wouldn't take a risky path and
    increased discount so the agent would be safer for higher rewards
    Also lowered noise for less randomness
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 1

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Every combo I put failed so I just returned NOT_POSSIBLE
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
