# added reward shaping and heuritic rollout
from template import Agent
import time
import math
import random
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
import Azul.azul_utils as utils

THINKTIME = 0.98
NUM_PLAYERS = 2

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)

    # Generates actions from this state with pruning
    def GetActions(self, state, agentID):
        actions = self.game_rule.getLegalActions(state, agentID)
        pruned_actions = []
        agent_state = state.agents[agentID]
        for action in actions:
            if action[0] in [utils.Action.TAKE_FROM_FACTORY, utils.Action.TAKE_FROM_CENTRE]:
                tile_grab = action[2]
                # Exclude actions that only fill the floor line unnecessarily
                if tile_grab.num_to_floor_line == tile_grab.number and any(
                    agent_state.lines_number[i] < i + 1 and
                    (agent_state.lines_tile[i] == -1 or agent_state.lines_tile[i] == tile_grab.tile_type)
                    for i in range(agent_state.GRID_SIZE)
                ):
                    continue
            pruned_actions.append(action)
        return pruned_actions

    # Carry out a given action on this state.
    def DoAction(self, state, action, agentID):
        state = self.game_rule.generateSuccessor(state, action, agentID)  
        return state

    # Heuristic policy for rollouts
    def HeuristicPolicy(self, state, agentID):
        actions = self.GetActions(state, agentID)
        # Prioritize actions based on heuristic scores
        best_actions = []
        best_score = -float('inf')
        for action in actions:
            score = self.EvaluateAction(state, action, agentID)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)

    # Evaluate an action based on heuristics
    def EvaluateAction(self, state, action, agentID):
        score = 0
        agent_state = state.agents[agentID]
        if action[0] in [utils.Action.TAKE_FROM_FACTORY, utils.Action.TAKE_FROM_CENTRE]:
            tile_grab = action[2]
            pattern_line = tile_grab.pattern_line_dest
            if pattern_line >= 0:
                tiles_needed = (pattern_line + 1) - agent_state.lines_number[pattern_line]
                tiles_added = min(tile_grab.number, tiles_needed)
                fill_ratio = (agent_state.lines_number[pattern_line] + tiles_added) / (pattern_line + 1)
                score += fill_ratio * 10  # Weight for pattern line completion
            else:
                # Penalize adding tiles to the floor line
                score -= tile_grab.num_to_floor_line * 5
        return score

    # Evaluate the score difference between self and opponent, with heuristics
    def EvaluateScore(self, game_state):
        
        # Introduce small reward for filling up middle columns???
        
        oppId = 1 if self.id == 0 else 0
        selfScore = game_state.agents[self.id].ScoreRound()[0]+game_state.agents[self.id].EndOfGameScore()
        oppScore = game_state.agents[oppId].ScoreRound()[0]+game_state.agents[oppId].EndOfGameScore()
        scoreEval = selfScore-oppScore
        return scoreEval

    def IsTerminalState(self, state):
        for plr_state in state.agents:
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows > 0:
                return True
        return False

    # Select an action using MCTS.
    def SelectAction(self, actions, game_state):
        start_time = time.time()
        root_node = self.Node(state=deepcopy(game_state), agent_id=self.id, agent=self)
        iteration = 0
        while time.time() - start_time < THINKTIME:
            iteration += 1
            node = root_node
            # Selection
            while not node.IsTerminalNode():
                if not node.IsFullyExpanded():
                    break
                node = node.SelectChild()
                if node is None:
                    break
            if node is None:
                continue  # Skip to next iteration
            # Expansion
            if not node.IsTerminalNode():
                if not node.IsFullyExpanded():
                    node = node.Expand()
            # Simulation
            reward = node.Rollout()
            # Backpropagation
            node.Backpropagate(reward)
        print(f"Iterations: {iteration}")
        # Choose the action leading to the most visited child node
        if root_node.children:
            best_child = max(root_node.children, key=lambda c: c.visits)
            best_action = best_child.action
        else:
            # If no children, select a random action
            best_action = random.choice(actions)
        return best_action

    class Node:
        def __init__(self, state, parent=None, action=None, agent_id=0, agent=None):
            self.state = state          # Game state at this node
            self.parent = parent        # Parent node
            self.action = action        # Action taken to get to this node
            self.children = []          # List of child nodes
            self.visits = 0             # Number of times node has been visited
            self.reward = 0.0           # Total reward from simulations
            self.untried_actions = []   # Actions not yet tried from this node
            self.agent_id = agent_id    # Agent to play at this node
            self.agent = agent          # Reference to the agent

            self.untried_actions = self.agent.GetActions(self.state, self.agent_id)

        def IsTerminalNode(self):
            return self.agent.IsTerminalState(self.state)

        def IsFullyExpanded(self):
            return len(self.untried_actions) == 0

        def SelectChild(self):
            # Use UCB1 formula to select a child node.
            if not self.children:
                return None
            c = 0.5  # Adjusted exploration parameter
            best_score = -float('inf')
            best_child = None
            for child in self.children:
                if child.visits == 0:
                    ucb = float('inf')
                else:
                    exploitation = child.reward / child.visits
                    exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                    ucb = exploitation + exploration
                if ucb > best_score:
                    best_score = ucb
                    best_child = child
            return best_child

        def Expand(self):
            # Expand one of the untried actions.
            if not self.untried_actions:
                return self  # Cannot expand further
            action = self.untried_actions.pop()
            next_state = deepcopy(self.state)
            next_state = self.agent.DoAction(next_state, action, self.agent_id)
            # Determine next agent_id.
            next_agent_id = (self.agent_id + 1) % NUM_PLAYERS
            child_node = self.agent.Node(next_state, parent=self, action=action, agent_id=next_agent_id, agent=self.agent)
            self.children.append(child_node)
            return child_node

        def Rollout(self):
            # Simulate the game until the end.
            current_state = deepcopy(self.state)
            current_agent_id = self.agent_id
            while not self.agent.IsTerminalState(current_state):
                actions = self.agent.GetActions(current_state, current_agent_id)
                if not actions:
                    break
                action = self.agent.HeuristicPolicy(current_state, current_agent_id)
                current_state = self.agent.DoAction(current_state, action, current_agent_id)
                # Update agent_id.
                current_agent_id = (current_agent_id + 1) % NUM_PLAYERS
            # At the end, compute final score.
            for agent in current_state.agents:
                agent.EndOfGameScore()
            reward = self.agent.EvaluateScore(current_state)
            return reward

        def Backpropagate(self, reward):
            # Update the node's statistics.
            self.visits += 1
            self.reward += reward
            if self.parent:
                self.parent.Backpropagate(reward)