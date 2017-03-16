"""
Game Agent
"""

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    # print('Timeout')
    pass



def custom_score(game, player):  #func: score_3opposition <-this one is my prefer play
    """
    Improved_score with a multiplier on the opposition moves
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Calculate my valid moves and the opponen moves.
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    result = float(len(my_moves)) - 3 * float(len(opponent_moves))
    return result


def score_agressive_defensive(game, player):  #func: 20170316 score_agressive_defensive
    """
    Improved_score with a multiplier on the opposition moves
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #state of the game
    state = float(len(game.get_blank_spaces())) / float(game.width * game.height)
 
    # Calculate my valid moves and the opponent moves.
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    if(state > 0.5): 
        #print("normal")
        result = float(len(my_moves) * 2 - len(opponent_moves))
    else:
        #print("aggressive")
        result = float(len(my_moves) - 2 * len(opponent_moves))
    
    return result




def score_20170311(game, player): #20170311
    """
    Calculate every position giving a weight based on the location
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Calculate factor to weight based on position
    factor = float(100 / game.width)
    op_factor = factor / 0.5

    # Calculate the middle of the board
    middle_x, middle_y = middle_node(game)

    # Calculate my valid moves and the opponen moves.
    my_moves = game.get_legal_moves(player)
    myself = 0
    for move in my_moves:
        myself += abs(middle_x - move[0]) * \
            factor + abs(middle_y - move[1]) * factor

    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent = 0
    for move in opponent_moves:
        opponent += abs(middle_x - move[0]) * \
            op_factor + abs(middle_y - move[1]) * op_factor

    result = myself - opponent

    return result


def score_weighted_status(game, player): #20170312 score_weighted_status
    """
    Calculate every position giving a weight based on the location
    Center gets 1, around 0.5 then 0.33, 025 as the distance get's
    bigger
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Calculate the middle of the board
    middle_x, middle_y = middle_node(game)

    # Calculate my valid moves and the opponen moves.
    my_moves = game.get_legal_moves(player)
    myself = 0
    for move in my_moves:
        value = max(abs(middle_x - move[0]), abs(middle_y - move[1]))
        factor = float(1 / float(value+1))
        myself += factor

    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent = 0
    for move in opponent_moves:
        value = max(abs(middle_x - move[0]), abs(middle_y - move[1]))
        factor = float(1 / float(value+1))
        opponent += factor

    result = myself - opponent

    return result



def middle_node(game):
    """
    Middle helper
    """
    return int(game.width/2) + 1, int(game.height/2) + 1

def near_middle_nodes(game):
    """
    L-Shapes helper
    """
    near_center = []
    around_middle = [(2, 1), (-2, -1), (-2, 1), (2, -1), (1, 2), (-1, -2), (-1, 2), (1, -2)]
    width, height = middle_node(game)

    for (row, column) in around_middle:
        near_center.append((width + row, height + column))

    return near_center


def score_simple_plus_current(game, player):  #func: 20170313 score_simple_plus_current
    """Variation using the idea of the L-shape surround as with a extra
    weight of 1.5 if it's the center and 0.5 if it's just the inmediate surrounding
    of the center and only considering the current position of my player
    for a given board
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Game is over, return
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #initial values
    weigth = 0

    # get the middle and nearby nodes
    middle = middle_node(game)
    near_middle = near_middle_nodes(game)

    # check the location of the player
    # and provide a weight
    # based on that
    cur_player = game.get_player_location(player)

    if cur_player in middle:
        weigth = 1.5

    if cur_player in near_middle:
        weigth = 0.5

    # calculate my moves and my opponent moves
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    # return the diferences of avaliable moves with weight
    return float(len(my_moves) - len(opponent_moves) + weigth)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=20.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        max_depth = 0
        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # Return inmediately if not legal moves available

        if not legal_moves:
            return (-1, -1)

        # Take the first move from the legal as the best just in case
        # I timed out
        b_score = float('-inf')
        b_move = legal_moves[0]

        # Create an Openining Book
        # Middle is better
        open_x = int(float(game.width) / 2)
        open_y = int(float(game.width) / 2)
        opening_book = [(open_x,  open_y), (open_x - 1, open_y), (open_x + 1,  open_y)]

        if game.move_count <= 1:
            return opening_book[game.move_count]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:
                # Interactive Deepening with Minimax / Alphabeta
                depth = 1
                while True:

                    if self.time_left() < self.TIMER_THRESHOLD:
                        raise Timeout()

                     # Do minimax or alphabeta
                    if self.method == 'minimax':
                        score, move = self.minimax(game, depth)
                    else:
                        score, move = self.alphabeta(game, depth)

                    if score > b_score:
                        b_move = move
                        b_score = score

                    depth += 1
            else:
                # Do minimax or alphabeta
                if self.method == 'minimax':
                    b_score, b_move = self.minimax(game, self.search_depth)
                else:
                    b_score, b_move = self.alphabeta(game, self.search_depth)

        except Timeout:
            #try:
            #    print(depth)
            #except NameError:
            #    print("-")
            
            return b_move

        # Return the best move
        return b_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Cutting point of recursion
        # retun
        if (depth == 0):
            score = self.score(game, self)
            move = game.get_player_location(game.inactive_player)
            return score, move

        # Get legal moves, if none return -1 -1
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return self.score(game, self), (-1, -1)

        # Move and check my child Nodes
        scores = []
        for move in legal_moves:
            c_score, c_move = self.minimax(game.forecast_move(
                move), depth - 1, not maximizing_player)
            scores.append([c_score, move])

        # Find the Best Score and Move
        b_score, b_move = scores[0]
        for score, move in scores:
            if maximizing_player:
                if score > b_score:
                    b_score = score
                    b_move = move
            else:
                if score < b_score:
                    b_score = score
                    b_move = move

        return b_score, b_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # best move is my current position
        if maximizing_player:
            b_score = float("-inf")
        else:
            b_score = float("inf")
        b_move = (-1, -1)

        legal_moves = game.get_legal_moves()

        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self), b_move

        for move in legal_moves:
            c_score, _ = self.alphabeta(
                game.forecast_move(move), depth - 1, alpha, beta, not maximizing_player)

            # Eval
            if maximizing_player:
                if c_score > b_score:
                    b_score = c_score
                    b_move = move

                # Prune
                if b_score >= beta:
                    break

                # Calculate new Alpha
                alpha = max(alpha, b_score)
            else:
                if c_score < b_score:
                    b_score = c_score
                    b_move = move

                # Prune
                if b_score <= alpha:
                    break

                # Calculate new Beta
                beta = min(beta, b_score)

        return b_score, b_move
