from openai import OpenAI
from dotenv import load_dotenv
import json
import random
load_dotenv()
# --- Configuration for the OpenAI client ---
# Replace with your actual API key and base URL if different
# Note: The API_KEY is left empty as per instructions, assuming it's handled by the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
AI_MODEL = "claude-3-5-sonnet-20241022" # The model specified by the user

# Initialize the OpenAI client
try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None # Set client to None if initialization fails

# --- Reversi Game Board and Logic (Simplified for demonstration) ---
# Represents the board: 0 = empty, 1 = Black (Player), 2 = White (AI)
# A typical Reversi board is 8x8.
BOARD_SIZE = 8
EMPTY = 0
BLACK = 1
WHITE = 2

def create_board():
    """Initializes a standard Reversi board."""
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # Starting pieces
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board

def print_board(board):
    """Prints the current state of the board."""
    print("\n  A B C D E F G H")
    for r_idx, row in enumerate(board):
        print(f"{r_idx+1} ", end="")
        for cell in row:
            if cell == BLACK:
                print("⚫", end=" ")
            elif cell == WHITE:
                print("⚪", end=" ")
            else:
                print(" .", end=" ")
        print()
    print("-" * 20)

def is_valid_move(board, row, col, player):
    """
    Checks if a move is valid for the given player at (row, col).
    This is a simplified check and does not implement full Reversi rules
    for flipping pieces, but rather focuses on identifying if *any* flip
    is possible in any direction.
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE) or board[row][col] != EMPTY:
        return False

    opponent = WHITE if player == BLACK else BLACK
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for dr, dc in directions:
        r, c = row + dr, col + dc
        found_opponent = False
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opponent:
            r, c = r + dr, c + dc
            found_opponent = True
        if found_opponent and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            return True # A valid flip sequence was found
    return False

def get_valid_moves(board, player):
    """Returns a list of all valid moves for the given player."""
    valid_moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_valid_move(board, r, c, player):
                valid_moves.append((r, c))
    return valid_moves

def make_move(board, row, col, player):
    """
    Makes a move and flips pieces. This is a simplified implementation
    and may not cover all complex Reversi flipping rules perfectly.
    """
    if not is_valid_move(board, row, col, player):
        return False, board # Move is not valid

    new_board = [row[:] for row in board] # Create a copy of the board
    new_board[row][col] = player

    opponent = WHITE if player == BLACK else BLACK
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    flipped_any = False
    for dr, dc in directions:
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and new_board[r][c] == opponent:
            to_flip.append((r, c))
            r, c = r + dr, c + dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and new_board[r][c] == player:
            # Found a closing piece, flip all in between
            for fr, fc in to_flip:
                new_board[fr][fc] = player
                flipped_any = True
    return flipped_any, new_board

def board_to_string(board):
    """Converts the board state to a string representation for the LLM."""
    s = ""
    for r_idx, row in enumerate(board):
        for cell in row:
            if cell == BLACK:
                s += "B" # Black piece
            elif cell == WHITE:
                s += "W" # White piece
            else:
                s += "." # Empty
        if r_idx < BOARD_SIZE - 1:
            s += "\n"
    return s

# --- Game State Management ---
game_board = create_board()
move_history = [] # Stores (player, move_coords, board_state_after_move)
current_player = BLACK # Black starts

# --- AI Reasoning Function ---
def get_ai_move_reasoning(board_state_str, difficulty_level, history, valid_moves_list):
    """
    Uses the OpenAI client (via Groq API) to get AI's reasoned move.
    """
    if client is None:
        return "AI is offline (client not initialized).", None

    # Convert valid moves list to a readable format for the LLM
    valid_moves_readable = [f"({chr(65+c)}, {r+1})" for r, c in valid_moves_list]

    prompt_messages = [
        {"role": "system", "content": f"""You are an expert Reversi (Othello) AI Grandmaster.
        Your goal is to choose the best move for the White player (W) on an {BOARD_SIZE}x{BOARD_SIZE} board.

        Current Board State (B=Black, W=White, .=Empty):
        {board_state_str}

        Previous Moves History (Player, Move, Board State After Move):
        {json.dumps(history)}

        Available Valid Moves for White: {', '.join(valid_moves_readable)}

        Difficulty Level: {difficulty_level}

        Think step-by-step about the best strategic move for White, considering:
        1. Maximizing your disc count.
        2. Minimizing opponent's disc count.
        3. Controlling corners and edges.
        4. Avoiding moves that give opponent access to corners.
        5. Creating stable discs.

        Output your reasoning and then the chosen move in the format:
        <reasoning>Your detailed reasoning here.</reasoning>
        <move>R,C</move> (where R is row 0-7, C is column 0-7)
        """},
        {"role": "user", "content": "What is the best next move for White?"}
    ]

    try:
        completion = client.chat.completions.create(
            messages=prompt_messages,
            model=AI_MODEL,
            max_tokens=512,
            temperature=0.5,
            top_p=0.7
        )

        ai_response_content = completion.choices[0].message.content

        # Parse reasoning and move from the LLM's response
        reasoning_start = ai_response_content.find("<reasoning>")
        reasoning_end = ai_response_content.find("</reasoning>")
        move_start = ai_response_content.find("<move>")
        move_end = ai_response_content.find("</move>")

        reasoning = "No reasoning found."
        chosen_move = None

        if reasoning_start != -1 and reasoning_end != -1:
            reasoning = ai_response_content[reasoning_start + len("<reasoning>"):reasoning_end].strip()

        if move_start != -1 and move_end != -1:
            move_str = ai_response_content[move_start + len("<move>"):move_end].strip()
            try:
                r, c = map(int, move_str.split(','))
                if (r, c) in valid_moves_list:
                    chosen_move = (r, c)
                else:
                    reasoning += f"\nWarning: AI suggested invalid move {r},{c}. Choosing random valid move."
            except ValueError:
                reasoning += f"\nWarning: AI suggested malformed move '{move_str}'. Choosing random valid move."

        # Fallback to a random valid move if AI doesn't provide a valid one
        if chosen_move is None and valid_moves_list:
            chosen_move = random.choice(ai_valid_moves) # Use ai_valid_moves here
            reasoning += f"\n(AI fallback: Randomly selected move {chosen_move} as primary choice was invalid or not provided.)"

        return reasoning, chosen_move

    except Exception as e:
        return f"Error getting AI reasoning: {e}", None

# --- Main Game Loop Simulation ---
def simulate_game_turn(user_move_coords, difficulty_level="Grandmaster"):
    """
    Simulates one turn of the Reversi game: user makes a move, then AI responds.
    Stores move history and uses AI for reasoning.
    """
    global game_board, move_history, current_player

    print("\n--- Current Board State ---")
    print_board(game_board)

    # 1. User's Move (Black)
    print(f"Player {current_player} (Black)'s turn.")
    user_row, user_col = user_move_coords

    if is_valid_move(game_board, user_row, user_col, BLACK):
        flipped, new_board = make_move(game_board, user_row, user_col, BLACK)
        if flipped:
            game_board = new_board
            move_history.append(("Black", (user_row, user_col), board_to_string(game_board)))
            print(f"Black (Player) placed a piece at ({chr(65+user_col)}, {user_row+1}).")
            print_board(game_board)
        else:
            print("Invalid move: No pieces flipped. Please choose a valid move.")
            return # End turn if user move was invalid (no flips)
    else:
        print("Invalid move: Not a valid position or no flips possible. Please choose a valid move.")
        return # End turn if user move was invalid

    # Check if game ended after user's move (no valid moves for next player)
    current_player = WHITE # Switch to AI's turn
    ai_valid_moves = get_valid_moves(game_board, WHITE)
    if not ai_valid_moves:
        print("No valid moves for White (AI). Skipping AI turn.")
        current_player = BLACK # Switch back to player if AI can't move
        player_valid_moves = get_valid_moves(game_board, BLACK)
        if not player_valid_moves:
            print("No valid moves for Black (Player) either. Game Over!")
            # Implement game end logic here (score calculation, etc.)
        return

    # 2. AI's Move (White)
    print(f"Player {current_player} (White - AI)'s turn.")
    board_str = board_to_string(game_board)

    print("AI is thinking...")
    ai_reasoning, ai_chosen_move = get_ai_move_reasoning(
        board_str,
        difficulty_level,
        move_history,
        ai_valid_moves
    )
    print(f"\nAI Reasoning ({difficulty_level} mode):\n{ai_reasoning}")

    if ai_chosen_move:
        ai_row, ai_col = ai_chosen_move
        flipped, new_board = make_move(game_board, ai_row, ai_col, WHITE)
        if flipped:
            game_board = new_board
            move_history.append(("White", (ai_row, ai_col), board_to_string(game_board)))
            print(f"White (AI) placed a piece at ({chr(65+ai_col)}, {ai_row+1}).")
            print_board(game_board)
        else:
            print(f"AI made a move at ({chr(65+ai_col)}, {ai_row+1}) but no pieces flipped. This indicates an issue with AI's move selection or game logic.")
    else:
        print("AI could not determine a valid move or client failed. Skipping AI turn.")

    # Switch back to player for the next turn
    current_player = BLACK

# --- Example Usage ---
if __name__ == "__main__":
    print("Starting Reversi Game Simulation.")
    print_board(game_board)

    # Example: User makes a move (e.g., row 2, col 4, which is C3 on a 0-indexed board)
    # Note: For a real game, you'd get user input for moves.
    # Here, we're hardcoding a few example turns.
    print("\n--- Turn 1 ---")
    simulate_game_turn(user_move_coords=(2, 4), difficulty_level="Rookie") # Player moves C3

    print("\n--- Turn 2 ---")
    simulate_game_turn(user_move_coords=(5, 2), difficulty_level="Intermediate") # Player moves F3

    print("\n--- Turn 3 ---")
    simulate_game_turn(user_move_coords=(2, 3), difficulty_level="Grandmaster") # Player moves D3

    print("\n--- Simulation Complete ---")
    print("Final Board State:")
    print_board(game_board)
    print("\nMove History:")
    for move in move_history:
        print(f"  {move[0]} moved to {move[1]}, Board:\n{move[2]}")
