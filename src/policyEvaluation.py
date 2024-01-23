
import numpy as np

class PolicyEvaluation(object):

    def __init__(self, stateSpace, actionSpace, transitionFunction, rewardFunction, theta):
        self.stateSpace=stateSpace
        self.actionSpace=actionSpace
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        self.theta=theta

    def __call__(self, policy):
        V={s: 0 for s in self.stateSpace}
        delta=np.Inf
        while delta > self.theta:
            delta = 0
            for s in self.stateSpace:
                v=V[s]
                V[s]=sum([policy[s][a]*sum([self.transitionFunction(s, a, sPrime)*(self.rewardFunction(s, a, sPrime)+V[sPrime]) for sPrime in self.stateSpace]) for a in policy[s].keys()])
                delta=max(delta, abs(V[s]-v))
        return V

