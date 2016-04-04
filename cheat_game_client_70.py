import random

from math import exp
from collections import defaultdict
from cheat_game_client_25 import Agent_25

from cheat_game_server import Game
from cheat_game_server import Player, Human
from cheat_game_server import Claim, Take_Card, Cheat, Call_Cheat
from cheat_game_server import Rank, Suit, Card, ActionEnum
from cheat_game_client import Agent, DemoAgent


# TODO add an array of cards that the opponent has and go through revealed cards to know what the opponent has/doesn't have
# also update opponent cheat probability

# TODO: if the top rank equals to cards we have and the amount of cards not equal to the rank is the same as the
# amount of our cards same as the rank, cheat with those cards!

def rank_exists(rank, cards):
    for card in cards:
        if card.rank == rank:
            return True
    return False


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


def get_best_honest_move(honest_moves):
    max_count = 0
    best_move = None
    for move in honest_moves:
        if isinstance(move, Claim) and move.count > max_count:
            max_count = move.count
            best_move = move
    return best_move


def move_exists(honest_moves, move_type):
    for move in honest_moves:
        if isinstance(move, move_type):
            return True
    return False


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

def count_rank_in_cards(cards, rank):
    rank_count = 0
    for card in cards:
        if card.rank == rank:
            rank_count += 1
    return rank_count


def get_furthest_cards(cards, cheat_rank, count):
    dist = defaultdict(int)
    for ind, card in enumerate(cards):
        dist[card] = cheat_rank.dist(card.rank)
    claim_cards = sorted(dist, key=dist.get, reverse=True)[:count]
    return claim_cards


class Agent_70(Agent):
    def __init__(self, name):
        super(Agent_70, self).__init__(name)
        # my recent claims
        self._my_claims = []
        # my claims history since the last time cheat was called
        self._my_claims_history = []
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
        self.cheat_prob = {"NO_MOVES": 0.4, "AVAIL_CLAIMS": 0.1}
        # probability to decide to call cheat
        self.call_cheat_prob = {1: 0.06, 2: 0.014, 3: 0.2, 4: 0.25}

        self.last_top_rank = None

        # did I cheat in my last move
        self._my_last_move = Take_Card()
        # my last move cards
        self._my_last_move_cards = create_my_cards_list(self.cards)
        # cards that the opponent know I had
        self._my_known_cards = []
        # opponent cards probability, by taken card
        self._opponent_card_probability = {}

        self._opponent_taken_cards_since_rank = {}

        for rank in Rank:
            self._opponent_card_probability[rank] = 1

        for rank in Rank:
            self._opponent_taken_cards_since_rank[rank] = 0

        self.first_move = True

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

        if self.last_top_rank is None:
            self.last_top_rank = self.game.get_initial_card().rank

        # if any one of the players called cheat, update known information
        if last_action == ActionEnum.CALL_CHEAT or isinstance(self._my_last_move, Call_Cheat):
            self.init_known_information(last_action, cards_revealed)

        # keep record of opponent actions
        if last_action == ActionEnum.MAKE_CLAIM or last_action == ActionEnum.TAKE_CARD:
            if last_action == ActionEnum.MAKE_CLAIM:
                self._opponent_claims.append(last_claim)
                self._opponent_actions.append(last_claim)
            elif last_action == ActionEnum.TAKE_CARD:
                self.opponent_took_card(self.table.top_rank())

        if self.need_to_call_cheat(deck_count, table_count, opponent_count, last_action, last_claim, honest_moves):
            move = get_call_cheat_move(honest_moves)
        elif table_count < 2 and len(self.cards) > 1:
            move = self.empty_table_cheat()
        else:
            if self.winning_cheat_exists() and random.random() < 0.6:
                    move = self.get_winning_cheat()
            else:
                move = self.get_best_move(deck_count, last_claim, honest_moves)

                if isinstance(move, Cheat):
                    if not self.cheat_possible():
                        move = None
                    elif move.count is None:
                        move = self.get_cheat(table_count)

                # if can't cheat, take a card
                if move is None:
                    if move_exists(honest_moves, Claim):
                        move = get_best_honest_move(honest_moves)
                    elif move_exists(honest_moves, Take_Card):
                        move = get_take_card_move(honest_moves)
                    else:
                        move = self.get_cheat(table_count, True)

        self.last_top_rank = self.table.top_rank()

        if isinstance(move, Claim) or isinstance(move, Cheat):
            self.add_to_table_known_cards(move)
            my_claim = Claim(move.rank, move.count)
            self._my_claims_history.append(my_claim)
            self.last_top_rank = move.rank

        if last_action == ActionEnum.MAKE_CLAIM:
            self._opponent_taken_cards_since_rank[last_claim.rank] = 0

        self._my_last_move_cards = create_my_cards_list(self.cards)
        self._my_last_move = move

        if self.first_move:
            self.first_move = False

        return move

    def init_known_information(self, last_action, cards_revealed):
        self._table_known_cards = []
        self._opponent_claims = []
        self._opponent_actions = []
        self._my_claims_history = []

        # in case opponent called cheat
        if last_action == ActionEnum.CALL_CHEAT:
            if isinstance(self._my_last_move, Cheat):
                # if I had cheat and the opponent called cheat he now know the cards I get
                for card in cards_revealed:
                    if card not in self._my_known_cards:
                        self._my_known_cards.append(card)
                    if card in self._opponent_cards:
                        self._opponent_cards.remove(card)
            else:
                self.init_opponent_card_probability()

                # Opponent call cheat was wrong, I know the opponent cards
                for card in cards_revealed:
                    if card in self._my_known_cards:
                        self._my_known_cards.remove(card)
                    if card not in self._opponent_cards:
                        self._opponent_cards.append(card)
        elif isinstance(self._my_last_move, Call_Cheat):
            if len(self._my_last_move_cards) != len(self.cards):
                # i was wrong calling cheat
                for card in self.cards:
                    if not self.is_card_in_last_move(card):
                        if card not in self._my_known_cards:
                            self._my_known_cards.append(card)
                        if card in self._opponent_cards:
                            self._opponent_cards.remove(card)
            else:
                self.init_opponent_card_probability()

                self._opponent_times_cheated += 1
                # I know opponent cards
                for card in cards_revealed:
                    if card not in self._opponent_cards:
                        self._opponent_cards.append(card)
                    if card in self._my_known_cards:
                        self._my_known_cards.remove(card)

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

    def winning_cheat_exists(self):
        cards_of_rank_above = count_rank_in_cards(self.cards, Rank.above(self.table.top_rank()))
        cards_of_rank_below = count_rank_in_cards(self.cards, Rank.below(self.table.top_rank()))
        winning_cheat_above = cards_of_rank_above == len(self.cards) - cards_of_rank_above
        winning_cheat_below = cards_of_rank_below == len(self.cards) - cards_of_rank_below

        if winning_cheat_above:
            count_cheating_cards = cards_of_rank_above
        elif winning_cheat_below:
            count_cheating_cards = cards_of_rank_below
        else:
            count_cheating_cards = 0

        if count_cheating_cards == 4:
            if random.random() < 0.3:
                return True
            else:
                return False
        elif count_cheating_cards == 3:
            if random.random() < 0.5:
                return True
            else:
                return False

        return winning_cheat_above or winning_cheat_below

    def get_winning_cheat(self):
        rank_above = Rank.above(self.table.top_rank())
        rank_below = Rank.below(self.table.top_rank())
        cards_of_rank_above = count_rank_in_cards(self.cards, rank_above)
        cards_of_rank_below = count_rank_in_cards(self.cards, rank_below)
        if cards_of_rank_above == len(self.cards) - cards_of_rank_above:
            cheat_cards = get_furthest_cards(self.cards, rank_above, cards_of_rank_above)
            return Cheat(cheat_cards, rank_above, cards_of_rank_above)
        else:
            cheat_cards = get_furthest_cards(self.cards, rank_below, cards_of_rank_below)
            return Cheat(cheat_cards, rank_below, cards_of_rank_below)



    def get_cheat(self, table_count, force_cheat = False):
        # TODO: if the opponent knows that i have 2 or more cards, lie about them in a certain probability,
        # and tell the truth in a certain probability in the next move
        # Cheat

        cheat_rank = self.get_cheat_rank()

        if not force_cheat and not self.cheat_possible():
            return None

        cheat_count = 1

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
        claim_cards = get_furthest_cards(self.cards, cheat_rank, cheat_count)
        return Cheat(claim_cards, cheat_rank, cheat_count)

    def get_best_move(self, deck_count, last_claim, honest_moves):
        scores = {}
        available_claim = False
        top_rank = self.table.top_rank()
        rank_above = Rank.above(top_rank)
        rank_below = Rank.below(top_rank)
        rank_below_history_count = 0
        rank_above_history_count = 0

        if move_exists(honest_moves, Claim):
            best_honest_move = get_best_honest_move(honest_moves)
            scores[best_honest_move] = best_honest_move.count + 0.5
            available_claim = True

        if move_exists(honest_moves, Take_Card):
            take_card_move = get_take_card_move(honest_moves)
            if available_claim:
                scores[take_card_move] = -1
            else:
                scores[take_card_move] = 0.6

        if are_cards_same_rank(self.cards):
            scores[Cheat()] = -1
        else:
            if available_claim:
                scores[Cheat()] = self.cheat_prob["AVAIL_CLAIMS"]
            else:
                scores[Cheat()] = self.cheat_prob["NO_MOVES"]

        # randomize scores add random in [-0.5..0.5)
        for move, score in scores.iteritems():
            scores[move] = score + 0.5 * (2.0 * random.random() - 1)

        # select move based on max score
        move = max(scores, key=scores.get)

        # at 80% don't lie if you have 4 same cards
        if isinstance(move, Claim):
            if move.count == 4 and random.random() < 0.8:
                return move
            elif move.count == 1:
                return move

        if are_cards_same_rank(self.cards):
            if move_exists(honest_moves, Claim):
                return get_best_honest_move(honest_moves)
            else:
                return get_take_card_move(honest_moves)

        # calculate how many cards I already claimed to put:
        for h_claim in self._my_claims_history:
            if h_claim.rank == rank_below:
                rank_below_history_count += h_claim.count
            if h_claim.rank == rank_above:
                rank_above_history_count += h_claim.count

        rank_above_score = rank_below_score = 0

        # choose cheat rank based on distance to remaining agent's card
        for card in self.cards:
            rank_above_score += card.rank.dist(rank_above)
            rank_below_score += card.rank.dist(rank_below)

        known_rank_above_score = known_rank_below_score = 0

        # choose cheat rank based on distance to remaining agent's card
        for card in self._my_known_cards:
            if card.rank == rank_above:
                known_rank_above_score += 1
            elif card.rank == rank_below:
                known_rank_below_score += 1

        # if I claimed already 3 or more and I still have 2 or more - I would like to make a honest claim with this rank
        if rank_above_history_count > 2 and rank_above_score > 1:
            move = get_best_honest_move(honest_moves)
        elif rank_above_history_count <= 2 and known_rank_above_score > 2:
            #if random.random() < 0.3:
                rank_count = count_rank_in_cards(self.cards, rank_above)
                cheat_count = max(known_rank_above_score - rank_above_history_count, rank_count)
                cheat_count = min(cheat_count, len(self.cards) - 1)
                move = self.known_cards_cheat(rank_above, cheat_count)
            #else:
            #    move = get_best_honest_move(honest_moves)
        elif rank_below_history_count > 2 and rank_below_score > 1:
            move = get_best_honest_move(honest_moves)
        elif rank_below_history_count <= 2 and known_rank_below_score > 2:
            #if random.random() < 0.3:
            rank_count = count_rank_in_cards(self.cards, rank_below)
            cheat_count = max(known_rank_below_score - rank_below_history_count, rank_count)
            cheat_count = min(cheat_count, len(self.cards) - 1)
            move = self.known_cards_cheat(rank_below, cheat_count)
        elif deck_count > 30 and move_exists(honest_moves, Claim):
            best_move = get_best_honest_move(honest_moves)
            if best_move.count == 2:
                if random.random() < 0.4:
                    claim_cards = get_furthest_cards(self.cards, best_move.rank, best_move.count)
                    move = Cheat(claim_cards, best_move.rank, best_move.count)
                else:
                    move = get_best_honest_move(honest_moves)


            #else:
            #    move = get_best_honest_move(honest_moves)

        # if I claimed 1 or 0 and I have in my known cards and in hand 2 or more - I would like to lie with 2 cards

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

        cards_count_after_whatif_honest_play = self.get_cards_ranks_count_after_whatif_honest_move(honest_moves)
        if cards_count_after_whatif_honest_play < 1:
            return False

        if not Rank.above(self.last_top_rank) == last_claim.rank and not Rank.below(self.last_top_rank) == last_claim.rank:
            if not self._silent: print "Impossible"
            return True

        # there is only 4 cards of each number,
        # so I know for sure that the opponent had cheat
        if rank_known_cards + last_claim.count > 4:
            if not self._silent: print "Who are you kidding?"
            return True

        opponent_cards_count_of_rank = 0
        for card in self._opponent_cards:
            if card.rank == last_claim.rank:
                opponent_cards_count_of_rank += 1

        already_claimed_rank = False
        opponent_taken_cards_since_same_claim = 0
        if len(self._opponent_actions) > 0:
            for opp_action in reversed(self._opponent_actions[:-1]):
                if isinstance(opp_action, Take_Card):
                    opponent_taken_cards_since_same_claim += 1
                elif isinstance(opp_action, Claim) and opp_action.rank == last_claim.rank and not already_claimed_rank:
                    already_claimed_rank = True
                    if opp_action.count != last_claim.count:
                        if deck_count + opponent_taken_cards_since_same_claim == 0:
                            if not self._silent: print "Deck Impossible"
                            return True

                        # count the take card count and check what's the probability
                        # the opponent will get this card (with number of cards)
                        ranks_not_mine = 4 - rank_known_cards
                        probability_that_matching_card_pulled = opponent_taken_cards_since_same_claim * ranks_not_mine / float(
                                deck_count + opponent_taken_cards_since_same_claim)
                        if random.random() > probability_that_matching_card_pulled + 0.25:
                            if not self._silent: print "Mismatch 1"
                            return True
                    else:
                        # opponent cheated before or cheats right now
                        # TODO: identify the cheat probability of the opponent
                        # if random.random() < self._opponent_cheat_probability:
                        if random.random() < 0.5:
                            if not self._silent: print "Fooled me once"
                            return True

        if not already_claimed_rank:

            # in case the opponent has the last card to complete the suit, but we think he doesn't have it
            if rank_known_cards + last_claim.count == 4 and last_claim.count == 1 and opponent_cards_count_of_rank < 1:
                probability_that_opponent_true = opponent_count / float(deck_count + opponent_count)

                # random the probability that the opponent has the cards
                if random.random() > probability_that_opponent_true + 0.15:
                    if not self._silent: print "Filling 4"
                    return True

            if opponent_cards_count_of_rank > 0:
                if opponent_cards_count_of_rank == last_claim.count:
                    return False
                else:

                    if random.random() < 0.15:
                        if not self._silent: print "Mismatch 2"
                        return True

            if self._opponent_card_probability[last_claim.rank] < 1:
                ranks_not_mine = 4 - rank_known_cards
                opponent_taken_cards_since_rank = self._opponent_taken_cards_since_rank[last_claim.rank]
                if deck_count + opponent_taken_cards_since_rank == 0:
                    return True

                    # probability_that_opponent_true = opponent_taken_cards_since_rank / float(deck_count + opponent_taken_cards_since_rank)
                    # if random.random() > probability_that_opponent_true + 0.1:
                    #    return True

            if last_claim.count == 1 and table_count <= 2:
                return False

            # define a probability to call cheat, based on deck count, opponent cards count, time opponent cheated,
            # table count, claim count

            # decaying function of number of cards on the table - call cheat less when risk is large
            cards_count_after_whatif_honest_play = self.get_cards_ranks_count_after_whatif_honest_move(honest_moves)
            table_ranks_count = self.get_table_ranks_count(table_count)
            decaying_function_param = table_ranks_count - cards_count_after_whatif_honest_play + opponent_count*2
            call_cheat = 0.25 * exp(-0.1 * decaying_function_param)

            if random.random() < call_cheat:
                if not self._silent: print "Decaying"
                return True

            if deck_count > 33 and opponent_count <= 10:
                if last_claim.count == 3:
                    if random.random() < 0.7:
                        if not self._silent: print "Deck"
                        return True
                elif last_claim.count == 2:
                    if random.random() < 0.2:
                        if not self._silent: print "Deck"
                        return True

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

    def get_cheat_rank(self):
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

        if cheat_rank == rank_below and count_rank_in_cards(self._opponent_cards, cheat_rank):
            cheat_rank = rank_above
        elif cheat_rank == rank_above and count_rank_in_cards(self._opponent_cards, cheat_rank):
            cheat_rank = rank_below

        return cheat_rank

    def empty_table_cheat(self):
        cheat_rank = self.get_cheat_rank()
        top_rank = self.table.top_rank()

        # Guess the cheat rank in a smart way
        if not rank_exists(top_rank, self.cards):
            if rank_exists(Rank.above(top_rank), self.cards):
                cheat_rank = Rank.above(top_rank)
            elif rank_exists(Rank.below(top_rank), self.cards):
                cheat_rank = Rank.below(top_rank)

        # in the worst case, be with one card left
        cheat_count = min(3, len(self.cards) - 1)

        # in case we really have 4 cards
        cheat_count = max(cheat_count, count_rank_in_cards(self.cards, cheat_rank))

        # select cards furthest from current claim rank
        claim_cards = get_furthest_cards(self.cards, cheat_rank, cheat_count)
        return Cheat(claim_cards, cheat_rank, cheat_count)

    def known_cards_cheat(self, cheat_rank, cheat_count):
        # select cards furthest from current claim rank
        claim_cards = get_furthest_cards(self.cards, cheat_rank, cheat_count)
        return Cheat(claim_cards, cheat_rank, cheat_count)

    def opponent_took_card(self, table_top_rank):
        self._opponent_actions.append(Take_Card())
        self._opponent_card_probability[Rank.above(table_top_rank)] = 0.1
        self._opponent_card_probability[Rank.below(table_top_rank)] = 0.1

        for rank in Rank:
            self._opponent_taken_cards_since_rank[rank] += 1

    def init_opponent_card_probability(self):
        for rank in Rank:
            self._opponent_card_probability[rank] = 1
            self._opponent_taken_cards_since_rank[rank] = 0

    def cheat_possible(self):
        top_rank = self.last_top_rank

        above_rank_count = count_rank_in_cards(self._opponent_cards, Rank.above(top_rank))
        below_rank_count = count_rank_in_cards(self._opponent_cards, Rank.above(top_rank))

        above_rank_possible = above_rank_count < 4
        below_rank_possible = below_rank_count < 4

        return above_rank_possible or below_rank_possible


agent_score = 0
agent_oldver_score = 0
for i in range(0, 1000):

    agent_oldver = DemoAgent("Demo Agent Old Ver.")
    #agent_oldver = Agent_25("agent 25")
    #agent_oldver = Agent_70("agent 25")
    agent_oldver.set_id(1)
    agent = Agent_70("Agent 70")
    agent.set_id(2)
    cheat = Game(agent_oldver, agent)
    cheat.play()

    if cheat.end_of_game():
        if cheat.winner == agent:
            agent_score += 1
        else:
            agent_oldver_score += 1
            break

        print '{0} wins: {1}'.format(agent_oldver.name, agent_oldver_score)
        print '{0} wins: {1}'.format(agent.name, agent_score)
