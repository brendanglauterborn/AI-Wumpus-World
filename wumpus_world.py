# Brendan Lauterborn
# Lab 2: Wumpus World
# sources:
# Cursor AI

import pygame, random, sys, os
import heapq

class WumpusWorld:
    def __init__(self, rows=4, cols=4):
        # Display settings
        self.cell_size = 120
        self.fps = 15

        # Colors
        self.bg = (30,30,30)
        self.wall = (60,60,60)
        self.open = (255,255,255)
        self.start_color = (128,128,128)
        self.visited_color = (120,120,120)

        # Grid values
        self.wall_value = 0
        self.open_value = 1
        self.pit_value = 2
        self.wumpus_value = 3
        self.gold_value = 4

        # World dimensions
        self.rows = rows
        self.cols = cols

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.cols * self.cell_size, self.rows * self.cell_size + 120))
        pygame.display.set_caption("Wumpus World")
        self.clock = pygame.time.Clock()

        # Game statistics tracking (initialize once, persist across resets)
        self.game_count = 1
        self.total_score = 0
        self.game_scores = []

        # Generate world
        self.reset_world()

    # ---------- utility: consistent marking ----------
    def mark_safe(self, cell):
        self.safe_cells.add(cell)
        self.dangerous_cells.discard(cell)
        self.possible_pits.discard(cell)
        self.possible_wumpus.discard(cell)

    def mark_dangerous(self, cell):
        self.dangerous_cells.add(cell)
        self.safe_cells.discard(cell)
        self.possible_pits.discard(cell)
        self.possible_wumpus.discard(cell)
    # -------------------------------------------------

    # ---------- helpers for movement safety ----------
    def front_cell(self):
        r, c = self.agent_pos
        dr, dc = [(0,1),(1,0),(0,-1),(-1,0)][self.agent_direction]
        return r + dr, c + dc

    def front_is_safe_strict(self):
        nr, nc = self.front_cell()
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False
        # Only step into PROVEN SAFE cells; never into candidates or dangerous
        if (nr, nc) in self.dangerous_cells: return False
        if (nr, nc) in self.possible_wumpus: return False
        return (nr, nc) in self.safe_cells

    def turn_toward_any_adjacent_safe(self):
        r, c = self.agent_pos
        for dir_idx, (dr, dc) in enumerate([(0,1),(1,0),(0,-1),(-1,0)]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if (nr, nc) in self.safe_cells and (nr, nc) not in self.possible_wumpus and (nr, nc) not in self.dangerous_cells:
                    if self.agent_direction != dir_idx:
                        return "turn_left" if (self.agent_direction - dir_idx) % 4 == 1 else "turn_right"
                    else:
                        return "move_forward"
        return None
    # -------------------------------------------------

    def track_game_score(self):
        """Track the final score for this game"""
        if not hasattr(self, 'game_scores'):
            self.game_scores = []
            self.total_score = 0
        
        # Only track if we haven't already tracked this game's score
        if len(self.game_scores) < self.game_count:
            self.game_scores.append(self.score)
            self.total_score += self.score
            print(f"Average score after {len(self.game_scores)} games: {self.total_score / len(self.game_scores):.1f}")

    def reset_world(self):
        """Generate a new Wumpus world"""
        # Track score if resetting during an active game (only if attributes exist)
        if hasattr(self, 'game_over') and hasattr(self, 'game_won'):
            if not self.game_over and not self.game_won:
                # If we're resetting during an active game, track the current score first
                self.track_game_score()
                # Increment game count after tracking the score
                self.game_count += 1
        
        self.grid, self.start, self.end, self.wumpus_cell, self.gold_cell = self.generate_wumpus_world()

        # Agent state
        self.agent_pos = self.start
        self.agent_direction = 0  # 0=East, 1=South, 2=West, 3=North
        self.has_arrow = True
        self.has_gold = False
        self.wumpus_alive = True
        self.game_over = False
        self.game_won = False

        # AI state
        self.visited_cells = {self.start}
        self.safe_cells = {self.start}
        self.dangerous_cells = set()
        self.possible_pits = set()
        self.possible_wumpus = set()
        self.action_timer = 0
        self.action_delay = 10

        # Add initial delay to see agent at start position
        self.initial_delay = 15  # 15 frames = ~1 second at 15 FPS

        # Seed initial safe neighbors if start position is safe
        self.mark_adjacent_cells_safe()

        # Performance measures
        self.score = 0
        self.actions_taken = 0

    def game_loop(self):
        """Main game loop"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_r:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        self.reset_world()

            # AI Agent Logic
            if not self.game_over and not self.game_won and self.action_timer <= 0 and self.initial_delay <= 0:
                planned = self.ai_agent_action()

                # === SAFETY BRAKE: never step forward into non-proven-safe ===
                if planned == "move_forward" and not self.front_is_safe_strict():
                    sensors_now = self.get_sensors()
                    nr, nc = self.front_cell()
                    # If we smell Stench and the cell ahead is a Wumpus candidate, prefer shooting
                    if "Stench" in sensors_now and self.has_arrow and (nr, nc) in self.possible_wumpus:
                        planned = "shoot"
                    else:
                        alt = self.turn_toward_any_adjacent_safe()
                        planned = alt or "turn_right"
                # ============================================================

                # Deduct 1 point for each action taken
                self.score -= 1
                self.actions_taken += 1

                if planned == "move_forward":
                    self.agent_pos, self.game_over = self.move_forward()
                    if not self.game_over:
                        self.visited_cells.add(self.agent_pos)
                        self.mark_safe(self.agent_pos)
                        # Mark adjacent cells as safe if no danger detected
                        self.mark_adjacent_cells_safe()
                    else:
                        # Agent died - deduct 1000 points
                        self.score -= 1000
                        # Track the final score for this game
                        self.track_game_score()
                        # Increment game count after tracking
                        self.game_count += 1
                elif planned == "turn_left":
                    self.agent_direction = (self.agent_direction - 1) % 4
                elif planned == "turn_right":
                    self.agent_direction = (self.agent_direction + 1) % 4
                elif planned == "grab":
                    self.has_gold = self.grab_gold()
                elif planned == "shoot":
                    wumpus_killed = not self.shoot_arrow()  # shoot_arrow returns False if wumpus killed
                    if wumpus_killed:
                        self.wumpus_alive = False
                        # Re-evaluate environment after killing wumpus
                        self.mark_adjacent_cells_safe()
                    else:
                        # If arrow missed and we're at start, climb out
                        if self.agent_pos == (0, 0):
                            self.game_won = True
                            if self.has_gold:
                                self.score += 1000
                            # Track the final score for this game
                            self.track_game_score()
                            # Increment game count after tracking
                            self.game_count += 1
                    self.has_arrow = False
                    # Deduct 10 points for using the arrow
                    self.score -= 10
                elif planned == "climb":
                    if self.agent_pos == self.start or self.actions_taken >= 40:
                        self.game_won = True
                        if self.has_gold:
                            self.score += 1000
                        # Track the final score for this game
                        self.track_game_score()
                        # Increment game count after tracking
                        self.game_count += 1

                self.action_timer = self.action_delay

            self.action_timer -= 1

            # Countdown initial delay
            if self.initial_delay > 0:
                self.initial_delay -= 1

            # Get sensor readings and draw
            sensors = self.get_sensors()
            self.draw_grid(sensors)
            pygame.display.flip()
            self.clock.tick(self.fps)

    def generate_wumpus_world(self):
        """Generate a new Wumpus world"""
        grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

        start = (0, 0)

        available_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) != start]

        # pits (~20% each)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != start and random.random() < 0.2:
                    grid[r][c].append('P')

        # wumpus
        wumpus_cell = random.choice(available_cells)
        grid[wumpus_cell[0]][wumpus_cell[1]].append('W')

        # gold
        gold_cell = random.choice(available_cells)
        grid[gold_cell[0]][gold_cell[1]].append('G')

        return grid, start, None, wumpus_cell, gold_cell

    def get_sensors(self):
        """Get sensor readings for the agent at the given position"""
        row, col = self.agent_pos
        sensors = []

        # Glitter (same tile)
        if 'G' in self.grid[row][col]:
            sensors.append("Glitter")

        # Stench (adjacent only)
        if self.wumpus_alive:
            for adj_row, adj_col in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
                if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                    if 'W' in self.grid[adj_row][adj_col]:
                        sensors.append("Stench")
                        break

        # Breeze (adjacent pit)
        for adj_row, adj_col in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
            if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                if 'P' in self.grid[adj_row][adj_col]:
                    sensors.append("Breeze")
                    break

        return sensors

    def move_forward(self):
        """Move agent forward in the current direction"""
        row, col = self.agent_pos
        new_row, new_col = row, col

        if self.agent_direction == 0:  # East
            new_col += 1
        elif self.agent_direction == 1:  # South
            new_row += 1
        elif self.agent_direction == 2:  # West
            new_col -= 1
        elif self.agent_direction == 3:  # North
            new_row -= 1

        # bounds check
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            # deadly?
            if 'P' in self.grid[new_row][new_col] or 'W' in self.grid[new_row][new_col]:
                return (new_row, new_col), True  # died
            return (new_row, new_col), False
        else:
            return self.agent_pos, False  # bump: stay put

    def grab_gold(self):
        row, col = self.agent_pos
        if 'G' in self.grid[row][col]:
            self.grid[row][col].remove('G')
            return True
        return False

    def shoot_arrow(self):
        """Shoot arrow in the current direction. Returns False if wumpus killed, True if missed"""
        row, col = self.agent_pos
        wumpus_row, wumpus_col = self.wumpus_cell

        hit = False
        
        # If shooting from start (0,0), arrow only travels east
        if row == 0 and col == 0 and self.agent_direction == 0:  # East from start
            # Check if wumpus is directly east of start
            if wumpus_row == 0 and wumpus_col == 1:
                hit = True
        else:
            # Original logic for other positions
            if self.agent_direction == 0:  # East
                if row == wumpus_row and col < wumpus_col:
                    hit = True
            elif self.agent_direction == 1:  # South
                if col == wumpus_col and row < wumpus_row:
                    hit = True
            elif self.agent_direction == 2:  # West
                if row == wumpus_row and col > wumpus_col:
                    hit = True
            elif self.agent_direction == 3:  # North
                if col == wumpus_col and row > wumpus_row:
                    hit = True

        if hit and self.wumpus_alive:
            # Wumpus killed!
            self.wumpus_alive = False
            # Remove W from the grid
            if 'W' in self.grid[wumpus_row][wumpus_col]:
                self.grid[wumpus_row][wumpus_col].remove('W')
            return False  # Wumpus was alive, now dead
        else:
            return True  # Missed (or wumpus already dead)

    def mark_adjacent_cells_safe(self):
        """Mark adjacent cells as safe if no breeze or stench is detected"""
        row, col = self.agent_pos
        sensors = self.get_sensors()


        if "Breeze" not in sensors and "Stench" not in sensors:
            adjacent_cells = self._adjacent_cells(row, col)
            newly_marked = []
            for adj_row, adj_col in adjacent_cells:
                if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                    # Don't mark as safe if it's already known to be dangerous
                    if ((adj_row, adj_col) not in self.visited_cells and 
                        (adj_row, adj_col) not in self.safe_cells and 
                        (adj_row, adj_col) not in self.dangerous_cells):
                        self.mark_safe((adj_row, adj_col))
                        newly_marked.append((adj_row, adj_col))
            if newly_marked:
                pass  # No new safe cells to mark
        else:
            pass  # Danger detected, not marking adjacent cells as safe

        self.perform_logical_inference()

    def perform_logical_inference(self):
        """Perform logical inference to identify pits and Wumpus locations"""
        visited_sensor_data = {}
        for vr, vc in self.visited_cells:
            sensors = self.simulate_sensors_at_position(vr, vc)
            visited_sensor_data[(vr, vc)] = sensors

        self.infer_pit_locations(visited_sensor_data)
        if self.wumpus_alive:
            self.infer_wumpus_locations(visited_sensor_data)

    def infer_pit_locations(self, visited_sensor_data):
        """Infer pit locations based on breeze patterns"""
        breeze_cells = [pos for pos, s in visited_sensor_data.items() if "Breeze" in s]

        all_possible_pits = set()
        for (row, col) in breeze_cells:
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                ar, ac = row+dr, col+dc
                if 0 <= ar < self.rows and 0 <= ac < self.cols and (ar, ac) not in self.visited_cells:
                    all_possible_pits.add((ar, ac))

        self.find_minimal_pit_explanation(breeze_cells, all_possible_pits)

        # Fallback: single-candidate for any breeze
        for (row, col) in breeze_cells:
            candidates = []
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                ar, ac = row+dr, col+dc
                if 0 <= ar < self.rows and 0 <= ac < self.cols and (ar, ac) not in self.visited_cells:
                    candidates.append((ar, ac))
            if len(candidates) == 1:
                pit_location = candidates[0]
                self.mark_dangerous(pit_location)
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    ar, ac = row+dr, col+dc
                    if 0 <= ar < self.rows and 0 <= ac < self.cols and (ar, ac) != pit_location:
                        if (ar, ac) not in self.visited_cells:
                            self.mark_safe((ar, ac))

        # Adjacent breeze pair shared cell
        for i, c1 in enumerate(breeze_cells):
            for c2 in breeze_cells[i+1:]:
                if abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) == 1:
                    shared = self.find_shared_adjacent_cell(c1, c2)
                    if shared and shared not in self.visited_cells:
                        self.mark_dangerous(shared)


    def find_shared_adjacent_cell(self, cell1, cell2):
        """Find the cell that is adjacent to both cell1 and cell2"""
        row1, col1 = cell1
        row2, col2 = cell2

        adjacent_to_cell1 = set()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row1 + dr, col1 + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                adjacent_to_cell1.add((new_row, new_col))

        adjacent_to_cell2 = set()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row2 + dr, col2 + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                adjacent_to_cell2.add((new_row, new_col))

        shared_cells = adjacent_to_cell1.intersection(adjacent_to_cell2)
        return list(shared_cells)[0] if shared_cells else None

    def find_minimal_pit_explanation(self, breeze_cells, all_possible_pits):
        """Find the minimal set of pits that explains all breezes"""
        from itertools import combinations

        if not breeze_cells:
            return

        all_valid = []
        for k in range(1, min(len(all_possible_pits)+1, 4)):
            for combo in combinations(all_possible_pits, k):
                explained = set()
                for pit in combo:
                    for b in breeze_cells:
                        if abs(pit[0]-b[0]) + abs(pit[1]-b[1]) == 1:
                            explained.add(b)
                if len(explained) == len(breeze_cells):
                    all_valid.append(combo)

        if not all_valid:
            return

        min_size = min(len(c) for c in all_valid)
        mins = [c for c in all_valid if len(c) == min_size]

        if len(mins) == 1:
            definite = mins[0]
            for pit in definite:
                self.mark_dangerous(pit)

            # Only mark cells as safe if NOT adjacent to any breeze cell
            for p in all_possible_pits:
                if p not in definite:
                    is_adjacent_to_breeze = any(abs(p[0]-b[0]) + abs(p[1]-b[1]) == 1 for b in breeze_cells)
                    if not is_adjacent_to_breeze:
                        self.mark_safe(p)
                    else:
                        pass  # Keeping as possible (adjacent to breeze)
        else:
            pass  # Multiple minimum-size solutions exist; keep candidates as possible only

    def infer_wumpus_locations(self, visited_sensor_data):
        """Triangulate Wumpus candidates strictly; only mark dangerous if unique"""
        if not self.wumpus_alive:
            self.possible_wumpus.clear()
            return

        stench_cells = {pos for pos, s in visited_sensor_data.items() if "Stench" in s}
        no_stench_cells = {pos for pos, s in visited_sensor_data.items() if "Stench" not in s}
        if not stench_cells:
            self.possible_wumpus.clear()
            return

        def adj(r, c):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield (nr, nc)

        # candidates must be adjacent to ALL stench cells
        candidates = None
        for (r, c) in stench_cells:
            nset = set(adj(r, c))
            candidates = nset if candidates is None else (candidates & nset)
        candidates = candidates or set()

        # remove any candidate adjacent to a no-stench cell
        banned = set()
        for (r, c) in no_stench_cells:
            banned |= set(adj(r, c))
        candidates -= banned

        # never pick start
        candidates.discard(self.start)

        self.possible_wumpus = set(candidates)

        if len(candidates) == 1:
            w = next(iter(candidates))
            self.mark_dangerous(w)
        else:
            pass  # Multiple possible Wumpus locations

    def _adjacent_cells(self, row, col):
        return [(row-1,col), (row+1,col), (row,col-1), (row,col+1)]

    def ai_agent_action(self):
        """Smart AI agent that uses logical reasoning based on sensors"""
        row, col = self.agent_pos
        sensors = self.get_sensors()

        self.perform_logical_inference()

        # Priority 0: Timeout - return to start after 40 moves if we can't make progress
        if self.actions_taken >= 40:
            # If not at start, try to return to start
            if self.agent_pos != self.start:
                return self.return_to_start_with_astar()
            else:
                # If already at start, climb out
                self.game_won = True
                if self.has_gold:
                    self.score += 1000
                return "climb"
        
        # Show move count for debugging
        if self.actions_taken % 10 == 0 and self.actions_taken > 0:
            pass  # Move count debug removed

        # Priority 1: If we have gold and are at start, climb out
        if self.has_gold and self.agent_pos == (0, 0):
            return "climb"

        # Priority 1.5: If we're at start and sense immediate danger, climb (but not if only stench)
        if self.agent_pos == (0, 0) and ("Breeze" in sensors or "Stench" in sensors):
            immediate = False
            for dr, dc in [(0,1),(1,0)]:  # E and S from (0,0)
                ar, ac = row+dr, col+dc
                if 0 <= ar < self.rows and 0 <= ac < self.cols:
                    if ('P' in self.grid[ar][ac]) or ('W' in self.grid[ar][ac] and self.wumpus_alive):
                        immediate = True
                        break
            # If there's a breeze (pit danger), always climb
            if "Breeze" in sensors and immediate:
                return "climb"
            # If there's only stench (wumpus danger), let Priority 1.6 handle it
            elif "Stench" in sensors and "Breeze" not in sensors:
                pass  # Only stench at start, letting shooting logic handle it
            elif immediate:
                return "climb"
            else:
                pass  # Danger sensed at start but no immediate threat, continuing

        # Priority 1.6: If we're at start with only stench, shoot arrow east
        if self.agent_pos == (0, 0) and "Stench" in sensors and "Breeze" not in sensors and self.has_arrow and self.wumpus_alive:
            # Make sure we're facing east
            if self.agent_direction != 0:
                return "turn_right" if self.agent_direction == 3 else "turn_left"
            else:
                return "shoot"

        # Priority 2: If we have gold, go home using A*
        if self.has_gold:
            return self.return_to_start_with_astar()

        # Priority 3: If glitter, grab
        if "Glitter" in sensors:
            return "grab"

        # Priority 4: Wumpus hunting only if blocking path to gold
        if self.has_arrow and "Stench" in sensors and self.wumpus_alive:
            w_loc = self.find_wumpus_location()
            if w_loc and self.is_wumpus_blocking_gold(w_loc):
                target_dir = self.get_direction_to_target(self.agent_pos, w_loc)
                if self.agent_direction != target_dir:
                    return "turn_left" if (self.agent_direction - target_dir) % 4 == 1 else "turn_right"
                else:
                    return "shoot"

        # Priority 5: Explore unvisited logically safe cells (but never into Wumpus candidates)
        unvisited_safe = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.visited_cells and self.is_cell_logically_safe(r, c) and (r, c) not in self.possible_wumpus:
                    unvisited_safe.append((r, c))

        if unvisited_safe:
            nearest = min(unvisited_safe, key=lambda cell: abs(cell[0]-row) + abs(cell[1]-col))
            path = self.a_star_pathfinding(self.agent_pos, nearest)
            if path and len(path) > 1:
                next_pos = path[1]
                cr, cc = self.agent_pos
                nr, nc = next_pos
                if nr < cr: target_dir = 3
                elif nr > cr: target_dir = 1
                elif nc < cc: target_dir = 2
                else: target_dir = 0
                if self.agent_direction != target_dir:
                    return "turn_left" if (self.agent_direction - target_dir) % 4 == 1 else "turn_right"
                else:
                    return "move_forward"

        # Priority 6: Very early exploration bias (still avoid Wumpus candidates)
        if len(self.visited_cells) <= 2:
            adj = []
            if row > 0: adj.append((row-1, col))
            if row < self.rows-1: adj.append((row+1, col))
            if col > 0: adj.append((row, col-1))
            if col < self.cols-1: adj.append((row, col+1))
            for ar, ac in adj:
                if (ar, ac) not in self.visited_cells and self.is_cell_logically_safe(ar, ac) and (ar, ac) not in self.possible_wumpus:
                    if ar < row: target_dir = 3
                    elif ar > row: target_dir = 1
                    elif ac < col: target_dir = 2
                    else: target_dir = 0
                    if self.agent_direction != target_dir:
                        return "turn_left" if (self.agent_direction - target_dir) % 4 == 1 else "turn_right"
                    else:
                        return "move_forward"

        # Fallback: strategic backtracking
        backtrack_action = self.strategic_backtracking()

        if backtrack_action == "turn_right" and len(self.visited_cells) > 2:
            sensors = self.get_sensors()
            if self.agent_pos == (0, 0) and "Breeze" not in sensors and "Stench" not in sensors:
                return "climb"
            elif len(self.visited_cells) >= 8:
                return self.return_to_start_with_astar()

        return backtrack_action

    def return_to_start_with_astar(self):
        path = self.a_star_pathfinding(self.agent_pos, self.start)
        if path and len(path) > 1:
            next_pos = path[1]
            cr, cc = self.agent_pos
            nr, nc = next_pos
            if nr < cr: target_dir = 3
            elif nr > cr: target_dir = 1
            elif nc < cc: target_dir = 2
            else: target_dir = 0
            if self.agent_direction != target_dir:
                return "turn_left" if (self.agent_direction - target_dir) % 4 == 1 else "turn_right"
            else:
                return "move_forward"
        return self.simple_backtrack_to_start()

    def a_star_pathfinding(self, start, goal):
        """A* pathfinding algorithm to find optimal path"""
        def neighbors(r, c):
            result = []
            for nr, nc in self._adjacent_cells(r, c):
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    # Only traverse if safe OR logically safe AND not a Wumpus candidate/danger
                    if ((nr, nc) in self.safe_cells) or (self.is_cell_logically_safe(nr, nc) and (nr, nc) not in self.possible_wumpus and (nr, nc) not in self.dangerous_cells):
                        result.append((nr, nc))
            return result

        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        cost = {start: 0}
        parent = {start: None}
        open_heap = []
        count = 0
        heapq.heappush(open_heap, (heuristic(start, goal), count, start))
        in_open = {start}

        while open_heap:
            _, _, (r, c) = heapq.heappop(open_heap)
            in_open.discard((r, c))
            if (r, c) == goal:
                break
            for nbr in neighbors(r, c):
                g = cost[(r, c)] + 1
                if nbr not in cost or g < cost[nbr]:
                    cost[nbr] = g
                    parent[nbr] = (r, c)
                    f = g + heuristic(nbr, goal)
                    if nbr not in in_open:
                        count += 1
                        heapq.heappush(open_heap, (f, count, nbr))
                        in_open.add(nbr)

        path = []
        if goal in parent:
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
        return path

    def simple_backtrack_to_start(self):
        r, c = self.agent_pos

        def is_ok(nr, nc):
            return (0 <= nr < self.rows and 0 <= nc < self.cols) and (
                (nr, nc) in self.safe_cells or (self.is_cell_logically_safe(nr, nc) and (nr, nc) not in self.possible_wumpus and (nr, nc) not in self.dangerous_cells)
            )

        candidates = [
            (r-1, c, 3),  # North
            (r+1, c, 1),  # South
            (r, c-1, 2),  # West
            (r, c+1, 0),  # East
        ]
        best = None
        best_d = abs(r) + abs(c)
        for nr, nc, d in candidates:
            if is_ok(nr, nc) and (abs(nr) + abs(nc) < best_d):
                best = (nr, nc, d)
                best_d = abs(nr) + abs(nc)

        if best:
            _, _, target_dir = best
            if self.agent_direction != target_dir:
                return "turn_left" if (self.agent_direction - target_dir) % 4 == 1 else "turn_right"
            else:
                return "move_forward"

        return "turn_right"

    def strategic_backtracking(self):
        best = self.find_best_exploration_position()
        if best:
            path = self.a_star_pathfinding(self.agent_pos, best)
            if path and len(path) > 1:
                next_pos = path[1]
                cr, cc = self.agent_pos
                nr, nc = next_pos
                if nr < cr: target_dir = 3
                elif nr > cr: target_dir = 1
                elif nc < cc: target_dir = 2
                else: target_dir = 0
                if self.agent_direction != target_dir:
                    return "turn_left" if (self.agent_direction - target_dir) % 4 == 1 else "turn_right"
                else:
                    return "move_forward"
        return "turn_right"

    def find_best_exploration_position(self):
        best_pos, best_score = None, -1
        for safe_cell in self.safe_cells:
            if safe_cell == self.agent_pos:
                continue
            unvisited_adjacent = 0
            r, c = safe_cell
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in self.visited_cells and self.is_cell_logically_safe(nr, nc) and (nr, nc) not in self.possible_wumpus:
                        unvisited_adjacent += 1
            if unvisited_adjacent > best_score:
                best_score = unvisited_adjacent
                best_pos = safe_cell
        return best_pos

    def is_cell_logically_safe(self, row, col):
        if (row, col) in self.safe_cells:
            return True
        if (row, col) in self.dangerous_cells:
            return False
        if (row, col) in self.possible_pits:
            return False
        if self.is_possible_wumpus_location(row, col):
            return False
        if self.wumpus_alive and (row, col) == self.wumpus_cell:
            return False
        return self.logical_safety_analysis(row, col)

    def logical_safety_analysis(self, row, col):
        # Trust prior inference first
        if (row, col) in self.dangerous_cells:
            return False
        if (row, col) in self.safe_cells:
            return True

        adjacent_visited = [(vr, vc) for (vr, vc) in self.visited_cells
                            if abs(row - vr) + abs(col - vc) == 1]

        if not adjacent_visited:
            return False

        # must be clean on ALL visited neighbors
        for vr, vc in adjacent_visited:
            sensors = self.simulate_sensors_at_position(vr, vc)
            if "Stench" in sensors or "Breeze" in sensors:
                return False

        return True

    def simulate_sensors_at_position(self, row, col):
        sensors = []
        if 'G' in self.grid[row][col]:
            sensors.append("Glitter")
        if self.wumpus_alive:
            for ar, ac in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
                if 0 <= ar < self.rows and 0 <= ac < self.cols:
                    if 'W' in self.grid[ar][ac]:
                        sensors.append("Stench")
                        break
        for ar, ac in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
            if 0 <= ar < self.rows and 0 <= ac < self.cols:
                if 'P' in self.grid[ar][ac]:
                    sensors.append("Breeze")
                    break
        return sensors

    def is_possible_wumpus_location(self, row, col):
        if not self.wumpus_alive:
            return False
        if (row, col) in self.safe_cells:
            return False
        return (row, col) in self.possible_wumpus

    def find_wumpus_location(self):
        """Use current candidates if available; otherwise fall back to triangulation on visited cells."""
        if not self.wumpus_alive:
            return None
        if len(self.possible_wumpus) == 1:
            return next(iter(self.possible_wumpus))
        if len(self.possible_wumpus) > 1:
            r, c = self.agent_pos
            return min(self.possible_wumpus, key=lambda t: abs(t[0]-r)+abs(t[1]-c))

        stench_cells, no_stench = [], []
        for vr, vc in self.visited_cells:
            s = self.simulate_sensors_at_position(vr, vc)
            if "Stench" in s: stench_cells.append((vr, vc))
            else: no_stench.append((vr, vc))

        if not stench_cells:
            for rr in range(self.rows):
                for cc in range(self.cols):
                    if (rr, cc) not in self.visited_cells:
                        return (rr, cc)
            return None

        poss = []
        for rr in range(self.rows):
            for cc in range(self.cols):
                if (rr, cc) in self.visited_cells:
                    continue
                if all(abs(rr-sr)+abs(cc-sc) == 1 for (sr, sc) in stench_cells) and \
                   all(abs(rr-nr)+abs(cc-nc) != 1 for (nr, nc) in no_stench):
                    poss.append((rr, cc))
        if not poss:
            return None
        r, c = self.agent_pos
        return min(poss, key=lambda t: abs(t[0]-r)+abs(t[1]-c))

    def is_wumpus_blocking_gold(self, wumpus_location):
        if not hasattr(self, 'gold_cell') or self.gold_cell is None:
            return True
        path = self.a_star_pathfinding_with_obstacles(self.agent_pos, self.gold_cell, [wumpus_location])
        return not path

    def a_star_pathfinding_with_obstacles(self, start, goal, obstacles):
        def neighbors(r, c):
            result = []
            for nr, nc in self._adjacent_cells(r, c):
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in obstacles and (
                        ((nr, nc) in self.safe_cells) or
                        (self.is_cell_logically_safe(nr, nc) and (nr, nc) not in self.possible_wumpus and (nr, nc) not in self.dangerous_cells)
                    ):
                        result.append((nr, nc))
            return result

        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        cost = {start: 0}
        parent = {start: None}
        open_heap = []
        count = 0
        heapq.heappush(open_heap, (heuristic(start, goal), count, start))
        in_open = {start}

        while open_heap:
            _, _, (r, c) = heapq.heappop(open_heap)
            in_open.discard((r, c))
            if (r, c) == goal:
                break
            for nbr in neighbors(r, c):
                g = cost[(r, c)] + 1
                if nbr not in cost or g < cost[nbr]:
                    cost[nbr] = g
                    parent[nbr] = (r, c)
                    f = g + heuristic(nbr, goal)
                    if nbr not in in_open:
                        count += 1
                        heapq.heappush(open_heap, (f, count, nbr))
                        in_open.add(nbr)

        path = []
        if goal in parent:
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
        return path

    def find_nearest_safe_cell(self):
        if not self.safe_cells:
            return None
        r, c = self.agent_pos
        return min(self.safe_cells, key=lambda s: abs(s[0]-r)+abs(s[1]-c))

    def get_direction_to_target(self, agent_pos, target_pos):
        row, col = agent_pos
        tr, tc = target_pos
        if tr < row: return 3
        elif tr > row: return 1
        elif tc < col: return 2
        else: return 0

    def get_cell_sensors(self, row, col):
        """Get sensor readings for a specific cell (for display purposes)"""
        sensors = []
        
        # Check for breeze (adjacent pits)
        for adj_row, adj_col in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
            if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                if 'P' in self.grid[adj_row][adj_col]:
                    sensors.append('b')  # Breeze
                    break
        
        # Check for stench (adjacent wumpus, only if wumpus is alive)
        if self.wumpus_alive:
            for adj_row, adj_col in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
                if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                    if 'W' in self.grid[adj_row][adj_col]:
                        sensors.append('s')  # Stench
                        break
        
        return sensors

    def get_element_color(self, element):
        """Get the appropriate color for a game element"""
        if element == 'G':  # Gold
            return (255, 255, 0)  # Yellow
        elif element == 'W':  # Wumpus
            return (255, 0, 0)  # Red
        elif element == 'P':  # Pit
            return (139, 0, 0)  # Dark red/maroon
        elif element == 'b':  # Breeze
            return (173, 216, 230)  # Light blue
        elif element == 's':  # Stench
            return (255, 165, 0)  # Orange
        else:  # Default (shouldn't happen)
            return (0, 0, 0)  # Black

    def draw_grid(self, sensors=None):
        self.screen.fill(self.bg)
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.start:
                    color = self.start_color
                elif (i, j) in self.visited_cells:
                    color = self.visited_color
                else:
                    color = self.open

                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100,100,100), rect, 1)

                font = pygame.font.SysFont(None, 24)
                if (i, j) == self.agent_pos:
                    direction_symbols = ['→', '↓', '←', '↑']
                    text = font.render(f"A{direction_symbols[self.agent_direction]}", True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
                else:
                    elements = self.grid[i][j]
                    cell_sensors = self.get_cell_sensors(i, j)
                    
                    # Combine grid elements with sensor indicators
                    all_items = elements + cell_sensors
                    
                    if all_items:
                        if len(all_items) == 1:
                            color = self.get_element_color(all_items[0])
                            text = font.render(all_items[0], True, color)
                            text_rect = text.get_rect(center=rect.center)
                            self.screen.blit(text, text_rect)
                        elif len(all_items) == 2:
                            color1 = self.get_element_color(all_items[0])
                            color2 = self.get_element_color(all_items[1])
                            text1 = font.render(all_items[0], True, color1)
                            text2 = font.render(all_items[1], True, color2)
                            text1_rect = text1.get_rect(center=(rect.centerx - 10, rect.centery))
                            text2_rect = text2.get_rect(center=(rect.centerx + 10, rect.centery))
                            self.screen.blit(text1, text1_rect)
                            self.screen.blit(text2, text2_rect)
                        elif len(all_items) == 3:
                            color1 = self.get_element_color(all_items[0])
                            color2 = self.get_element_color(all_items[1])
                            color3 = self.get_element_color(all_items[2])
                            text1 = font.render(all_items[0], True, color1)
                            text2 = font.render(all_items[1], True, color2)
                            text3 = font.render(all_items[2], True, color3)
                            text1_rect = text1.get_rect(center=(rect.centerx, rect.centery - 8))
                            text2_rect = text2.get_rect(center=(rect.centerx - 8, rect.centery + 8))
                            text3_rect = text3.get_rect(center=(rect.centerx + 8, rect.centery + 8))
                            self.screen.blit(text1, text1_rect)
                            self.screen.blit(text2, text2_rect)
                            self.screen.blit(text3, text3_rect)
                        elif len(all_items) == 4:
                            color1 = self.get_element_color(all_items[0])
                            color2 = self.get_element_color(all_items[1])
                            color3 = self.get_element_color(all_items[2])
                            color4 = self.get_element_color(all_items[3])
                            text1 = font.render(all_items[0], True, color1)
                            text2 = font.render(all_items[1], True, color2)
                            text3 = font.render(all_items[2], True, color3)
                            text4 = font.render(all_items[3], True, color4)
                            text1_rect = text1.get_rect(center=(rect.centerx - 8, rect.centery - 8))
                            text2_rect = text2.get_rect(center=(rect.centerx + 8, rect.centery - 8))
                            text3_rect = text3.get_rect(center=(rect.centerx - 8, rect.centery + 8))
                            text4_rect = text4.get_rect(center=(rect.centerx + 8, rect.centery + 8))
                            self.screen.blit(text1, text1_rect)
                            self.screen.blit(text2, text2_rect)
                            self.screen.blit(text3, text3_rect)
                            self.screen.blit(text4, text4_rect)

        self.draw_hud(sensors)

    def draw_hud(self, sensors=None):
        font = pygame.font.SysFont(None, 18)
        y_offset = self.rows * self.cell_size + 10

        if hasattr(self, 'initial_delay') and self.initial_delay > 0:
            delay_text = f"Starting in {self.initial_delay // 15 + 1} seconds..."
            text = font.render(delay_text, True, (255,255,0))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

        if sensors:
            sensor_text = "Sensors: " + ", ".join(sensors)
            text = font.render(sensor_text, True, (255,255,255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

        if self.game_over:
            text = font.render("GAME OVER - Agent died!", True, (255,0,0))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
        elif self.game_won:
            if self.has_gold:
                text = font.render("YOU WIN! - Agent escaped with gold!", True, (0,255,0))
            else:
                text = font.render("YOU WIN! - Agent escaped safely!", True, (0,255,0))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

        inventory = []
        if self.has_arrow:
            inventory.append("Arrow")
        if self.has_gold:
            inventory.append("Gold")
        inventory_text = "Inventory: " + ", ".join(inventory) if inventory else "Inventory: Empty"
        text = font.render(inventory_text, True, (255,255,255))
        self.screen.blit(text, (10, y_offset))
        y_offset += 20

        score_text = f"Score: {self.score} | Actions: {self.actions_taken}"
        text = font.render(score_text, True, (255,255,255))
        self.screen.blit(text, (10, y_offset))
        y_offset += 20

        # Show average score if we have completed games
        if hasattr(self, 'game_scores') and len(self.game_scores) > 0:
            avg_score = self.total_score / len(self.game_scores)
            avg_text = f"Avg Score: {avg_score:.1f} after {len(self.game_scores)} games"
            text = font.render(avg_text, True, (255,255,0))  # Yellow color for average
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
        else:
            pass  # No games completed yet

        controls_text = "AI Agent Running - R=Reset, ESC=Quit"
        text = font.render(controls_text, True, (255,255,255))
        self.screen.blit(text, (10, y_offset))

def main():
    world = WumpusWorld(4, 4)
    world.game_loop()

if __name__ == "__main__":
    main()
