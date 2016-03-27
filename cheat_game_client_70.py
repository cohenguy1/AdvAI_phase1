import random
from math import exp
from collections import defaultdict

from cheat_game_server import Game
from cheat_game_server import Player, Human
from cheat_game_server import Claim, Take_Card, Cheat, Call_Cheat
from cheat_game_server import Rank, Suit, Card, ActionEnum
from cheat_game_client import Agent


# TODO add an array of cards that the opponent has and go through revealed cards to know what the opponent has/doesn't have
# also update opponent cheat probability

def are_cards_same_rank(cards):
    if len(cards) <= 1:
        return True

    first_card_rank = cards[0].rank
    for card in cards:
        if card.rank != first_card_rank:
            return False
    return True


def get_call_cheat_move(honest_moves):
    for move in honest_moves:
        if isinstance(move, Call_Cheat):
            return move

def get_take_card_move(honest_moves):
    for move in honest_moves:
        if isinstance(move, Take_Card):
            return move


def create_my_cards_list(cards):
    my_cards = []
    for card in cards:
        my_cards.append(Card(card.rank, card.suit))
    return my_cards


def ranks_count(cards):
    ranks = {}

    for card in cards:
        if ranks.get(card.rank):
            ranks[card.rank] += 1
        else:
            ranks[card.rank] = 1

    return len(ranks)


class Agent_70(Agent):
    def __init__(self, name):
        super(Agent_70, self).__init__(name)
        # my recent claims
        self._my_claims = []
        # cards that the opponents
        self._opponent_claims = []
        # opponent actions
        self._opponent_actions = []
        # cards who was at my hand and the opponent took
        self._opponent_cards = []
        # cards that I have discard
        self._table_known_cards = []
        # number of times the opponent cheated
        self._opponent_times_cheated = 0
        # probability of opponent cheat
        self._opponent_cheat_probability = 0
        # probability to decide to cheat
        self.cheat_prob = {"NO_MOVES": 0.33, "AVAIL_CLAIMS": 0.1}
        # probability to decide to call cheat
        self.call_cheat_prob = {1: 0.06, 2: 0.014, 3: 0.2, 4: 0.25}
        # did I cheat in my last move
        self._my_last_move = Take_Card()
        # my last move cards
        self._my_last_move_cards = create_my_cards_list(self.cards)
        # cards that the opponent know I had
        self._my_known_cards = []
        # number of times we try to cheat in a row and the opponent call cheat
        self._stuck_counter = 0

    """
    This function implements action logic / move selection for the agent\n
    :param deck_count: amount of cards in the deck
    :param table_count: amount of cards on the table
    :param opponent_count: amount of cards in the opponent hand
    :param last_action: the opponent last action from ActionEnum.TAKE_CARD or .MAKE_CLAIM or .CALL_CHEAT
    :param last_claim: the last claim of card discarding
    :param honest_moves: a list of available actions, other than making a false ("cheat") claim
    :param cards_revealed: if last action was "call cheat" cards on table were revealed
    :return: Action object Call_Cheat or Claim or Take_Card or Cheat
    """

    def agent_logic(self, deck_count, table_count, opponent_count,
                    last_action, last_claim, honest_moves, cards_revealed):

        if last_action == ActionEnum.CALL_CHEAT:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        if table_count < 2 and self._stuck_counter < 200:
            move = self.empty_table_cheat()


        # if any one of the players called cheat, update known information
        if last_action == ActionEnum.CALL_CHEAT or isinstance(self._my_last_move, Call_Cheat):
            self.init_known_information(last_action, cards_revealed)
        elif isinstance(self._my_last_move, Take_Card):
            self.add_new_card_to_my_list()
        if not move:
            move = self.get_best_move(last_claim, honest_moves)

            if isinstance(move, Cheat):
                move = self.get_cheat(table_count)

            # if can't cheat take a card
            if move == 0:
                move = get_take_card_move(honest_moves)

        if isinstance(move, Claim) or isinstance(move, Cheat):
            self.add_to_table_known_cards(move)

        # keep record of opponent actions
        if last_action == ActionEnum.MAKE_CLAIM or last_action == ActionEnum.TAKE_CARD:
            if last_action == ActionEnum.MAKE_CLAIM:
                self._opponent_claims.append(last_claim)
                self._opponent_actions.append(last_claim)
            elif last_action == ActionEnum.TAKE_CARD:
                self._opponent_actions.append(Take_Card())

        if deck_count == 0 and isinstance(move, Take_Card):
            move = get_call_cheat_move(honest_moves)

        self._my_last_move = move

        return move

    def init_known_information(self, last_action, cards_revealed):
        self._table_known_cards = []
        self._opponent_claims = []
        self._opponent_actions = []

        # in case opponent called cheat
        if last_action == ActionEnum.CALL_CHEAT:
            if isinstance(self._my_last_move, Cheat):
                # if I had cheat and the opponent called cheat he now know the cards I get
                for card in cards_revealed:
                    self._my_known_cards.append(card)
            else:
                # Opponent call cheat was wrong, I know the opponent cards
                for card in cards_revealed:
                    self._opponent_cards.append(card)
        elif isinstance(self._my_last_move, Call_Cheat):
            if len(self._my_last_move_cards) != len(self.cards):
                # i was wrong calling cheat
                for card in self.cards:
                    if not self.is_card_in_last_move(card):
                        self._my_known_cards.append(card)
            else:
                self._opponent_times_cheated += 1
                # I know opponent cards
                for card in self._cards_revealed:
                    self._opponent_cards.append(card)

    def add_to_table_known_cards(self, move):
        card_count = 0
        # adding to table known cards the cards that I really played
        for card in self.cards:
            if card.rank == move.rank and (card not in self._table_known_cards) and card_count < move.count:
                self._table_known_cards.append(card)
                for last_move_card in self._my_last_move_cards:
                    if card.rank == last_move_card.rank and card.suit == last_move_card.suit:
                        self._my_last_move_cards.remove(last_move_card)
                card_count += 1

    def add_new_card_to_my_list(self):
        for card in self.cards:
            if not self.is_card_in_last_move(card):
                self._my_last_move_cards.append(card)


    def is_card_in_last_move(self, card):
        card_found = False
        for last_move_card in self._my_last_move_cards:
            if card.rank == last_move_card.rank and card.suit == last_move_card.suit:
                card_found = True
        return card_found

    def get_cheat(self, table_count):
        # TODO: always lie on 3 cards when table is empty (distant cards)
        # TODO: cheat on previous claim of 2 ( 3 is large) at the start of the game
        # TODO: if the opponent knows that i have 2 or more cards, lie about them in a certain probability,
        # and tell the truth in a certain probability in the next move
        # Cheat
        top_rank = self.table.top_rank()
        rank_above = Rank.above(top_rank)
        rank_below = Rank.below(top_rank)
        rank_above_score = rank_below_score = 0

        # choose cheat rank based on distance to remaining agent's card
        for card in self.cards:
            rank_above_score += card.rank.dist(rank_above)
            rank_below_score += card.rank.dist(rank_below)
        if rank_above_score < rank_below_score:
            cheat_rank = rank_above
        else:
            cheat_rank = rank_below
        cheat_count = 1

        rank_above_table_cards = 0
        # for card in self._opponent_claims:
        #     if card.rank == rank_above:
        #         rank_above_table_cards += 1
        for card in self._opponent_cards:
            if card.rank == rank_above:
                rank_above_table_cards += 1

        rank_below_table_cards = 0
        # for card in self._opponent_claims:
        #     if card.rank == rank_above:
        #         rank_below_table_cards += 1
        for card in self._opponent_cards:
            if card.rank == rank_above:
                rank_below_table_cards += 1

        if (rank_below_table_cards > 2 and rank_above_table_cards < 3) or rank_below_table_cards == 4:
            cheat_rank = rank_above
        elif (rank_above_table_cards > 2 and rank_below_table_cards < 3) or rank_above_table_cards == 4:
            cheat_rank = rank_below

        if rank_below_table_cards == 4 and rank_above_table_cards == 4:
            return 0

        # decaying function of number of cards on the table - cheat less when risk is large
        r = 0.5 * exp(-0.1 * table_count)

        # choose cheat count
        while cheat_count < 4 and random.random() < r and len(self.cards) > (cheat_count + 1):
            cheat_count += 1

        opponent_rank = 0
        for card in self._opponent_cards:
            if cheat_rank == card.rank:
                opponent_rank += 1
        # if the opponent have 2 cards or more from the rank I cheat on, I would cheat with only one card
        if opponent_rank > 1:
            cheat_count = 1

        # select cards furthest from current claim rank
        dist = defaultdict(int)
        for ind, card in enumerate(self.cards):
            dist[card] = cheat_rank.dist(card.rank)
        claim_cards = sorted(dist, key=dist.get)[:cheat_count]
        return Cheat(claim_cards, cheat_rank, cheat_count)

    def get_best_move(self, last_claim, honest_moves):
        scores = {}
        available_claim = False

        for move in honest_moves:
            if isinstance(move, Claim):
                scores[move] = move.count
                available_claim = True
            elif isinstance(move, Take_Card):
                scores[move] = 0.6

        if are_cards_same_rank(self.cards):
            # TODO: always cheat when table is empty!
            # TODO: lie on 3 or less, leaving 1 card to discard
            scores[Cheat()] = -0.5
        else:
            if available_claim:
                scores[Cheat()] = self.cheat_prob["AVAIL_CLAIMS"]
            else:
                scores[Cheat()] = self.cheat_prob["NO_MOVES"]

        # randomize scores add random \in [-0.5..0.5)
        for move, score in scores.iteritems():
            scores[move] = score + 0.5 * (2.0 * random.random() - 1)


        # select move based on max score
        move = max(scores, key=scores.get)
        return move


    # TODO: at the beginning of the game (by the deck size), if the opponent is claiming for 2 or more,
    # call cheat at a certain percentile
    def need_to_call_cheat(self, deck_count, table_count, opponent_count, last_action, last_claim, honest_moves):
        if not last_action == ActionEnum.MAKE_CLAIM:
            return False
        if opponent_count == 0:
            return True

        rank_known_cards = 0
        for card in self._table_known_cards:
            if card.rank == last_claim.rank:
                rank_known_cards += 1

        for card in self.cards:
            if card.rank == last_claim.rank:
                rank_known_cards += 1

        # there is only 4 cards of each number,
        # so I know for sure that the opponent had cheat
        if rank_known_cards + last_claim.count > 4:
            return True

        opponent_cards_count_of_rank = 0
        for card in self._opponent_cards:
            if card.rank == last_claim.rank:
                opponent_cards_count_of_rank += 1

        # in case the opponent has the last card to complete the suit, but we think he doesn't have it
        if rank_known_cards + last_claim.count == 4 and last_claim.count == 1 and opponent_cards_count_of_rank < 1:
            probability_that_opponent_true = opponent_count / float(deck_count + opponent_count)

            # random the probability that the opponent has the cards
            if random.random() > probability_that_opponent_true + 0.1:
                return True

        already_claimed_rank = False
        opponent_taken_cards_since_same_claim = 0
        if len(self._opponent_actions) > 0:
            for opp_action in reversed(self._opponent_actions):
                if isinstance(opp_action, Take_Card):
                    opponent_taken_cards_since_same_claim += 1
                elif isinstance(opp_action, Claim) and opp_action.rank == last_claim.rank:
                    already_claimed_rank = True
                    if opp_action.count != last_claim.count:
                        # count the take card count and check what's the probability the opponent will get this card (with number of cards)
                        probability_that_matching_card_pulled = opponent_taken_cards_since_same_claim / float(
                                deck_count + opponent_taken_cards_since_same_claim)
                        if random.random() > probability_that_matching_card_pulled + 0.1:
                            return True
                    else:
                        # opponent cheated before or cheats right now
                        # TODO: identify the cheat probability of the opponent
                        if random.random() < self._opponent_cheat_probability:
                            return True

        if not already_claimed_rank:
            # define a probability to call cheat, based on deck count, opponent cards count, time opponent cheated,
            # table count, claim count

            # decaying function of number of cards on the table - call cheat less when risk is large
            cards_count_after_whatif_honest_play = self.get_cards_ranks_count_after_whatif_honest_move(honest_moves)
            table_ranks_count = self.get_table_ranks_count(table_count)
            decaying_function_param = table_ranks_count - cards_count_after_whatif_honest_play + opponent_count + self._opponent_times_cheated
            call_cheat = 0.33 * exp(-0.1 * decaying_function_param)

            if random.random() < call_cheat:
                return True

        all_prev_opponent_rank_claim_cards = 0
        for card in self._opponent_claims:
            if card.rank == last_claim.rank:
                all_prev_opponent_rank_claim_cards += 1

        return False

    def get_cards_ranks_count_after_whatif_honest_move(self, honest_moves):
        current_ranks_count = ranks_count(self.cards)

        if len(honest_moves) == 0:
            return current_ranks_count

        cards_count_after_move = {}
        for move in honest_moves:
            if isinstance(move, Claim):
                cards_count_after_move[move] = current_ranks_count - 1
            elif isinstance(move, Take_Card):
                cards_count_after_move[move] = current_ranks_count + 1

        if len(cards_count_after_move) == 0:
            return current_ranks_count

        best_move = min(cards_count_after_move, key=cards_count_after_move.get)
        min_cards_ranks_count = cards_count_after_move[best_move]
        return min_cards_ranks_count

    def call_cheat_was_correct(self, cards_revealed):
        first_card = cards_revealed[0]
        i_have_card_revealed = False
        for card in self.cards:
            if card.rank == first_card.rank and card.suit == first_card.suit:
                i_have_card_revealed = True
        return not i_have_card_revealed

    def get_table_ranks_count(self, table_count):
        table_ranks_count_my_claims = ranks_count(self._table_known_cards)
        table_count_my_claims = len(self._table_known_cards)

        # subtract my claims and add the ranks count that i really played
        table_count_without_my_claims = table_count - table_count_my_claims + table_ranks_count_my_claims
        return table_count_without_my_claims


    """
    if we have cards close to the card we are going to put from both sides
    we better put only one of that card because the other will be useful later
    """
    def is_to_separate(self, move):
        is_close_up = False
        is_close_down = False
        # choose cheat rank based on distance to remaining agent's card
        for card in self.cards:
            if card.rank - move.rank > 0 and card.rank - move.rank < 4:
                is_close_up = True
            if move.rank - card.rank > 0 and move.rank - card.rank < 4:
                is_close_down = True
        if is_close_up and is_close_down:
            return True
        return False

    def empty_table_cheat(self):
        top_rank = self.table.top_rank()
        rank_above = Rank.above(top_rank)
        rank_below = Rank.below(top_rank)
        rank_above_score = rank_below_score = 0

        # choose cheat rank based on distance to remaining agent's card
        for card in self.cards:
            rank_above_score += card.rank.dist(rank_above)
            rank_below_score += card.rank.dist(rank_below)
        if rank_above_score < rank_below_score:
            cheat_rank = rank_above
        else:
            cheat_rank = rank_below
        cheat_count = 3

        # select cards furthest from current claim rank
        dist = defaultdict(int)
        for ind, card in enumerate(self.cards):
            dist[card] = cheat_rank.dist(card.rank)
        claim_cards = sorted(dist, key=dist.get)[:cheat_count]
        return Cheat(claim_cards, cheat_rank, cheat_count)

cheat = Game(Agent_70("Demo 1"), Agent_70("me"))
cheat.play()
