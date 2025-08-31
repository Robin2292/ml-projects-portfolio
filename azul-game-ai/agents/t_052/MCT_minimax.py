# use MCT in the early game and minimax in the late game
from template import Agent
import time
import math
import random
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
import Azul.azul_utils as utils

THINKTIME = 0.975  # Time limit for thinking
NUM_PLAYERS = 2   # Number of players

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)
    
    # Generate possible actions
    def GetActions(self, game_state, agent_id):
        actions = []
        action_tracker = []

        if not game_state.TilesRemaining() and not game_state.next_first_agent == -1:
            return ["ENDROUND"]
        elif agent_id == self.game_rule.num_of_agent:
            return ["STARTROUND"]
        else: 
            agent_state = game_state.agents[agent_id]

            # Traverse pattern lines in priority order
            for i in [2,1,0,3,4]:
                fid = 0
                for fd in game_state.factories:
                    for tile in utils.Tile:
                        num_avail = fd.tiles[tile]
                        if num_avail == 0:
                            continue
                    
                        # Check if the tile can be placed on the pattern line
                        if agent_state.lines_tile[i] != -1 and \
                            agent_state.lines_tile[i] != tile:
                            continue
                        
                        grid_col = int(agent_state.grid_scheme[i][tile])
                        if agent_state.grid_state[i][grid_col] == 1:
                            continue

                        slots_free = (i+1) - agent_state.lines_number[i]
                        tg = utils.TileGrab()
                        tg.number = num_avail
                        tg.tile_type = tile
                        tg.pattern_line_dest = i
                        tg.num_to_pattern_line = min(num_avail, slots_free)
                        tg.num_to_floor_line = tg.number - tg.num_to_pattern_line

                        actions.append((utils.Action.TAKE_FROM_FACTORY, fid, tg))
                        action_tracker.append([fid, tile])
                    
                    fid += 1

                # Check the center pool
                for tile in utils.Tile:
                    
                    grid_col = int(agent_state.grid_scheme[i][tile])
                    if agent_state.grid_state[i][grid_col] == 1:
                        continue
                    
                    if agent_state.lines_tile[i] != -1 and \
                        agent_state.lines_tile[i] != tile:
                            continue

                    num_avail = game_state.centre_pool.tiles[tile]

                    if num_avail == 0:
                        continue 

                    slots_free = (i+1) - agent_state.lines_number[i]
                    tg = utils.TileGrab()
                    tg.number = num_avail
                    tg.tile_type = tile
                    tg.pattern_line_dest = i
                    tg.num_to_pattern_line = min(num_avail, slots_free)
                    tg.num_to_floor_line = tg.number - tg.num_to_pattern_line

                    action_tracker.append([-1, tile])
                    actions.append((utils.Action.TAKE_FROM_CENTRE, -1, tg))
           
            # Place tiles into the floor line
            for tile in utils.Tile:
                
                if [-1, tile] in action_tracker:
                    continue

                num_avail = game_state.centre_pool.tiles[tile]

                if num_avail == 0:
                    continue
                
                tg = utils.TileGrab()
                tg.number = num_avail
                tg.tile_type = tile
                tg.num_to_floor_line = tg.number
                actions.append((utils.Action.TAKE_FROM_CENTRE, -1, tg))

            fid = 0
            for fd in game_state.factories:
                
                if [fid, tile] in action_tracker:
                    continue
                
                for tile in utils.Tile:
                    num_avail = fd.tiles[tile]
                    if num_avail == 0:
                        continue
                
                    tg = utils.TileGrab()
                    tg.number = num_avail
                    tg.tile_type = tile
                    tg.num_to_floor_line = tg.number
                    actions.append((utils.Action.TAKE_FROM_FACTORY, fid, tg))
                
                fid += 1

            return actions

    def GetSortedActions(self, game_state, maximisingPlayer):
        agent_id = self.id if maximisingPlayer else (self.id + 1) % NUM_PLAYERS
        actions = self.GetActions(game_state, agent_id)
        scored_actions = []

        for action in actions:
            next_state = deepcopy(game_state)
            child = self.DoAction(next_state, action, agent_id)
            score = self.EvaluateScore(child)
            scored_actions.append((score, action))

        scored_actions.sort(reverse=maximisingPlayer, key=lambda x: x[0])

        return [action for _, action in scored_actions]
    
    # Perform action and return the new state
    def DoAction(self, state, action, agentID):
        state = self.game_rule.generateSuccessor(state, action, agentID)  
        return state

    # Evaluation function with reward shaping
    def EvaluateScore(self, game_state):
        
        # Introduce small reward for filling up middle columns???
        
        oppId = 1 if self.id == 0 else 0
        selfScore = game_state.agents[self.id].ScoreRound()[0]+game_state.agents[self.id].EndOfGameScore()
        oppScore = game_state.agents[oppId].ScoreRound()[0]+game_state.agents[oppId].EndOfGameScore()
        scoreEval = selfScore-oppScore
        return scoreEval

    def IsTerminalState(self, state):
        # Check if the state is a terminal state
        for plr_state in state.agents:
            if plr_state.GetCompletedRows() > 0:
                return True
        return False

    # Estimate the game phase based on the number of tiles placed
    def GetGamePhase(self, game_state):
        my_agent_state = game_state.agents[self.id]
        tiles_on_grid = sum([sum(row) for row in my_agent_state.grid_state])
        if tiles_on_grid <= 5:
            return 'early'
        else:
            return 'late'

    # Use different algorithms based on the game phase
    def SelectAction(self, actions, game_state):
        start_time = time.time()
        game_phase = self.GetGamePhase(game_state)

        if game_phase == 'early':
            # Use MCTS algorithm
            root_node = self.Node(state=game_state, parent=None, action=None, agent_id=self.id, agent=self)

            # MCTS implementation
            while time.time() - start_time < THINKTIME:
                node = root_node
                # Selection phase
                while not node.IsTerminalNode():
                    if not node.IsFullyExpanded():
                        break
                    node = node.SelectChild()
                    if node is None:
                        break
                if node is None:
                    continue
                # Expansion phase
                if not node.IsFullyExpanded():
                    node = node.Expand()
                # Rollout phase: simulate using reward shaping
                reward = node.Rollout()
                node.Backpropagate(reward)

            if root_node.children:
                best_child = max(root_node.children, key=lambda c: c.visits)
                best_action = best_child.action
            else:
                best_action = random.choice(actions)

            return best_action
        else:
            best_move = actions[0]
            depth = 1
            previous_sorted_moves = None

            while time.time() - start_time < THINKTIME:
                global NODECOUNT
                NODECOUNT = 0

                if not previous_sorted_moves:
                    previous_sorted_moves = self.GetSortedActions(game_state, True)

                reason, _, move = self.minimaxDepth(game_state, depth, True, -float('inf'), float('inf'), start_time, previous_sorted_moves)

                if reason != "time":
                    best_move = move

                depth += 1 

            return best_move
            


    def minimaxDepth(self, game_state, depth, maximisingPlayer, alpha, beta, start_time, previous_sorted_moves=None):
        global NODECOUNT
        NODECOUNT += 1

        if depth == 0:
            return "depth", self.EvaluateScore(game_state), None

        elif "ENDROUND" in self.GetActions(game_state, self.id):
            return "fulldepth", self.EvaluateScore(game_state), None

        elif time.time() - start_time > THINKTIME:
            return "time", None, None

        if maximisingPlayer:
            return self.MaximumBestChildAlphaBeta(game_state, depth, alpha, beta, start_time, previous_sorted_moves)
        else:
            return self.MinimumBestChildAlphaBeta(game_state, depth, alpha, beta, start_time, previous_sorted_moves)

    def MaximumBestChildAlphaBeta(self, game_state, depth, alpha, beta, start_time, previous_sorted_moves):
        actions = previous_sorted_moves if previous_sorted_moves else self.GetActions(game_state, self.id)
        maxEval = -float('inf')
        best_action = actions[0]

        for action in actions:
            next_state = deepcopy(game_state)
            child = self.DoAction(next_state, action, self.id)
            reason, eval, _ = self.minimaxDepth(child, depth - 1, False, alpha, beta, start_time)

            if reason == "time":
                return reason, maxEval, best_action

            if eval > maxEval:
                maxEval = eval
                best_action = action

            alpha = max(alpha, maxEval)
            if beta <= alpha:
                break

        return "fulldepth", maxEval, best_action

    def MinimumBestChildAlphaBeta(self, game_state, depth, alpha, beta, start_time, previous_sorted_moves):
        agentID = (self.id + 1) % NUM_PLAYERS
        actions = previous_sorted_moves if previous_sorted_moves else self.GetActions(game_state, agentID)
        minEval = float('inf')
        best_action = actions[0]

        for action in actions:
            next_state = deepcopy(game_state)
            child = self.DoAction(next_state, action, agentID)
            reason, eval, _ = self.minimaxDepth(child, depth - 1, True, alpha, beta, start_time)

            if reason == "time":
                return reason, minEval, best_action

            if eval < minEval:
                minEval = eval
                best_action = action

            beta = min(beta, minEval)
            if beta <= alpha:
                break  

        return "fulldepth", minEval, best_action

    # 
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
            return len(self.untried_actions) == 0

        def SelectChild(self):
            if not self.children:
                return None
            c = 1.4  # exploration factor
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
            action = self.untried_actions.pop()
            next_state = deepcopy(self.state)
            next_state = self.agent.DoAction(next_state, action, self.agent_id)
            next_agent_id = (self.agent_id + 1) % NUM_PLAYERS
            child_node = self.agent.Node(next_state, self, action, next_agent_id, self.agent)
            self.children.append(child_node)
            return child_node

        def Rollout(self):
            current_state = deepcopy(self.state)
            current_agent_id = self.agent_id
            rollout_depth = 5  # limit simulation depth
            for _ in range(rollout_depth):
                if self.agent.IsTerminalState(current_state):
                    break
                actions = self.agent.GetActions(current_state, current_agent_id)
                if not actions:
                    break
                action = random.choice(actions)
                current_state = self.agent.DoAction(current_state, action, current_agent_id)
                current_agent_id = (current_agent_id + 1) % NUM_PLAYERS
            reward = self.agent.EvaluateScore(current_state)
            return reward

        def Backpropagate(self, reward):
            self.visits += 1
            self.reward += reward
            if self.parent:
                self.parent.Backpropagate(reward)