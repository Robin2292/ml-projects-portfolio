# Azul AI Agent Project

**Course:** COMP90054 AI Planning for Autonomy  
**Project Type:** Group Project (2 students)  
**Topic:** Implementing an autonomous agent to play the board game Azul.  

---

## Project Goal
Develop an **autonomous Azul agent** that can play competitively in the **UoM Azul Contest**. The agent should use **at least three AI-related techniques** from the course or independent research, combined into a strong game-playing strategy.

---

## Techniques (at least 3 required)
Candidate approaches include:
1. Blind or heuristic search algorithms (with Azul-specific heuristics).  
2. Classical planning (e.g., PDDL).  
3. Policy iteration or value iteration (model-based MDP).  
4. Monte Carlo Tree Search (MCTS) or UCT (model-free MDP).  
5. Reinforcement learning (classical, approximate, or deep Q-learning).  
6. Goal recognition (to infer opponents’ intentions).  
7. Game theoretic methods.  

> Note: Hand-coded decision trees are allowed but **do not count** towards the 3 required AI techniques.

---

## Deliverables
- `agents/<t_XXX>/myTeam.py`: The AI agent implementation.  
- A **Wiki report**: documenting approaches, comparisons of different agents/techniques, and analysis of strengths and weaknesses.  

---

## Rules
- Code must run **error-free on Python 3.8+**.  
- Agents should never crash, and must perform pre-computation within the allowed 15 seconds before each game.  
- Good software engineering practices are required: clear commits, meaningful messages, collaboration via GitHub team tools.  

---

## Assessment
- **Code performance:** Tournament evaluation of the Azul agent.  
- **Wiki report:** Critical analysis, comparison of techniques, strengths & weaknesses.  
- **Teamwork & contribution:** Professional SE practices and balanced workload.  

---

## References
- [Azul Game Rules](https://www.ultraboardgames.com/azul/game-rules.php)  