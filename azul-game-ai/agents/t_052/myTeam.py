from template import Agent
import time
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
import Azul.azul_utils as utils

THINKTIME = 0.975
NUM_PLAYERS = 2

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)
    
    # Generates actions from this state.
    def GetActions(self, game_state, agent_id):
        actions = []
        action_tracker = []

        if not game_state.TilesRemaining() and not game_state.next_first_agent == -1:
            return ["ENDROUND"]
        elif agent_id == self.game_rule.num_of_agent:
            return ["STARTROUND"]
        else: 
            agent_state = game_state.agents[agent_id]

            # Look at each pattern line in order of strength
            for i in [2,1,0,3,4]:
                # Next, look at each factory display with available tiles
                fid = 0
                for fd in game_state.factories:
                    # Look at each available tile set
                    for tile in utils.Tile:
                        num_avail = fd.tiles[tile]
                        if num_avail == 0:
                            continue
                
                        # Can tiles be added to pattern line i?
                        if agent_state.lines_tile[i] != -1 and \
                            agent_state.lines_tile[i] != tile:
                            # these tiles cannot be added to this pattern line
                            continue
                        
                        # Is the space on the grid for this tile already
                        # occupied?
                        grid_col = int(agent_state.grid_scheme[i][tile])
                        if agent_state.grid_state[i][grid_col] == 1:
                            # It is, so we cannot place this tile type
                            # in this pattern line!
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

                # Check the centre pool
                for tile in utils.Tile:
                    
                    grid_col = int(agent_state.grid_scheme[i][tile])
                    if agent_state.grid_state[i][grid_col] == 1:
                        # It is, so we cannot place this tile type
                        # in this pattern line!
                        continue
                    
                    # Can tiles be added to pattern line i?
                    if agent_state.lines_tile[i] != -1 and \
                        agent_state.lines_tile[i] != tile:
                            # these tiles cannot be added to this pattern line
                            continue

                    # Number of tiles of this type in the centre
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
           
            # Alternately, add to the floor line
            for tile in utils.Tile:
                
                # Only add to floor line if you can't add to pattern line
                if [-1, tile] in action_tracker:
                    continue

                # Number of tiles of this type in the centre
                num_avail = game_state.centre_pool.tiles[tile]

                if num_avail == 0:
                    continue
                
                # Default action is to place all the tiles in the floor line
                tg = utils.TileGrab()
                tg.number = num_avail
                tg.tile_type = tile
                tg.num_to_floor_line = tg.number
                actions.append((utils.Action.TAKE_FROM_CENTRE, -1, tg))

            fid = 0
            for fd in game_state.factories:
                
                # Only add to the floor line if you can't add to a pattern line
                if [fid, tile] in action_tracker:
                    continue
                
                # Look at each available tile set
                for tile in utils.Tile:
                    num_avail = fd.tiles[tile]
                    if num_avail == 0:
                        continue
                
                    # Place all the tiles in the floor line
                    tg = utils.TileGrab()
                    tg.number = num_avail
                    tg.tile_type = tile
                    tg.num_to_floor_line = tg.number
                    actions.append((utils.Action.TAKE_FROM_FACTORY, fid, tg))
                
                fid += 1

            return actions
    
    def GetSortedActions(self, game_state, maximisingPlayer):
        actions = self.GetActions(game_state, self.id if maximisingPlayer else 1 - self.id)
        scored_actions = []

        for action in actions:
            next_state = deepcopy(game_state)
            child = self.DoAction(next_state, action, self.id if maximisingPlayer else 1 - self.id)
            score = self.EvaluateScore(child)
            scored_actions.append((score, action))

        scored_actions.sort(reverse=maximisingPlayer, key=lambda x: x[0])

        return [action for _, action in scored_actions]
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action, agentID):
        state = self.game_rule.generateSuccessor(state, action, agentID)  
        return state

    def CheckExpiry(self, start_time):
        if (time.time()-start_time < THINKTIME):
            return False
        return True
    
    # Implementing minimax, similar to the top chess engine Stockfish
    
    def SelectAction(self, actions, game_state):
        
        start_time = time.time()
        best_move = actions[0]
        depth = 1
        previous_sorted_moves = None

        while time.time() - start_time < THINKTIME:
            global NODECOUNT
            NODECOUNT = 0

            if not previous_sorted_moves:
                previous_sorted_moves = self.GetSortedActions(game_state, True)

            # Perform a Minimax search with sorted moves from the previous depth
            reason, _, move = self.minimaxDepth(game_state, depth, True, -float('inf'), float('inf'), start_time, previous_sorted_moves)

            if reason != "time":
                best_move = move  # Update best move if search didn't time out

            #print("Depth: ", depth, "Nodecount: ", NODECOUNT, "Reason: ", reason)
            depth += 1  # Increase the depth for the next iteration

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
                break  # Prune remaining branches

        return "fulldepth", maxEval, best_action

    def MinimumBestChildAlphaBeta(self, game_state, depth, alpha, beta, start_time, previous_sorted_moves):
        agentID = 1 if self.id == 0 else 0
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
                break  # Prune remaining branches

        return "fulldepth", minEval, best_action
    
    def EvaluateScore(self, game_state):
        
        # Introduce small reward for filling up middle columns???
        
        oppId = 1 if self.id == 0 else 0
        selfScore = game_state.agents[self.id].ScoreRound()[0]+game_state.agents[self.id].EndOfGameScore()
        oppScore = game_state.agents[oppId].ScoreRound()[0]+game_state.agents[oppId].EndOfGameScore()
        scoreEval = selfScore-oppScore
        return scoreEval
