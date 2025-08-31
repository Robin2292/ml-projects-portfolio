from template import Agent
import time
import math
import random
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule

THINKTIME = 0.98  # Time limit for making decisions
NUM_PLAYERS = 2

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)

    def GetActions(self, state, agentID):
        # Returns all legal actions for the agent at the given state
        return self.game_rule.getLegalActions(state, agentID)

    def DoAction(self, state, action, agentID):
        # Updates the game state by executing the given action by the agent
        state = self.game_rule.generateSuccessor(state, action, agentID)
        return state

    def EvaluateScore(self, game_state):
        my_score = game_state.agents[self.id].score
        opp_id = (self.id + 1) % NUM_PLAYERS
        opp_score = game_state.agents[opp_id].score
        # Factor in the number of legal actions to add strategic depth
        return my_score - opp_score + 0.5 * len(self.GetActions(game_state, self.id))

    def IsTerminalState(self, state):
        # if the current state is a terminal state
        for plr_state in state.agents:
            if plr_state.GetCompletedRows() > 0:
                return True
        return False

    def SelectAction(self, actions, game_state):
        # Selects the best action
        start_time = time.time()
        root_node = self.Node(state=game_state, parent=None, action=None, agent_id=self.id, agent=self)

        # Monte Carlo Tree Search (MCTS) implementation
        while time.time() - start_time < THINKTIME:
            node = root_node
            # Node selection phase
            while not node.IsTerminalNode():
                if not node.IsFullyExpanded():
                    break
                node = node.SelectChild()
                if node is None:
                    break
            if node is None:
                continue
            # Node expansion phase
            if not node.IsFullyExpanded():
                node = node.Expand()
            # Simulation phase: Simulate the outcome from the current game state
            reward = node.Rollout()
            # Backpropagation phase: Update the nodes with the simulation results
            node.Backpropagate(reward)

        # Decide the action based on the most visited node after the search
        if root_node.children:
            best_child = max(root_node.children, key=lambda c: c.visits)
            best_action = best_child.action
        else:
            best_action = random.choice(actions)

        return best_action

    class Node:
        def __init__(self, state, parent, action, agent_id, agent):
            self.state = state
            self.parent = parent
            self.action = action
            self.agent_id = agent_id
            self.agent = agent
            self.children = []
            self.untried_actions = self.agent.GetActions(state, agent_id)
            self.reward = 0
            self.visits = 0

        def IsTerminalNode(self):
            return self.agent.IsTerminalState(self.state)

        def IsFullyExpanded(self):
            # Checks if all possible actions at this node have been explored
            return len(self.untried_actions) == 0

        def SelectChild(self):
            # Selects the next child node to explore using the Upper Confidence Bound (UCB) formula
            if not self.children:
                return None
            c = 1.4  # Exploration factor for UCB
            best_score = -float('inf')
            best_child = None
            for child in self.children:
                exploitation = child.reward / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb = exploitation + exploration
                if ucb > best_score:
                    best_score = ucb
                    best_child = child
            return best_child

        def Expand(self):
            # Expands a new child node from an unexplored action
            action = self.untried_actions.pop()
            next_state = deepcopy(self.state)
            next_state = self.agent.DoAction(next_state, action, self.agent_id)
            next_agent_id = (self.agent_id + 1) % NUM_PLAYERS
            child_node = self.agent.Node(next_state, self, action, next_agent_id, self.agent)
            self.children.append(child_node)
            return child_node

        def Rollout(self):
            # Simulates a random playthrough from the current node to a terminal state
            current_state = deepcopy(self.state)
            current_agent_id = self.agent_id
            while not self.agent.IsTerminalState(current_state):
                actions = self.agent.GetActions(current_state, current_agent_id)
                if not actions:
                    break
                action = random.choice(actions)
                current_state = self.agent.DoAction(current_state, action, current_agent_id)
                current_agent_id = (current_agent_id + 1) % NUM_PLAYERS
            for agent in current_state.agents:
                agent.EndOfGameScore()
            reward = self.agent.EvaluateScore(current_state)
            return reward

        def Backpropagate(self, reward):
            # Updates this node and its ancestors with the reward from the rollout
            self.visits += 1
            self.reward += reward
            if self.parent:
                self.parent.Backpropagate(reward)