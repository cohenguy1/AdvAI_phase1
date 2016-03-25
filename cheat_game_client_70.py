import random
from math import exp
from collections import defaultdict

from cheat_game_server import Game
from cheat_game_server import Player, Human
from cheat_game_server import Claim, Take_Card, Cheat, Call_Cheat
from cheat_game_server import Rank, Suit, Card, ActionEnum

from cheat_game_client import Agent

class Agent_70(Agent):
    def __init__(self, name):
        super(Agent_70, self).__init__(name)
        # my recent claims
        self._my_claims = []
        # cards that the opponents
        self._opponent_claims = []
        # cards who was at my hand and the opponent took
        self._opponent_cards = []
        # cards that I have discard
        self._table_known_cards = []
        # probability to decide to cheat
        self.cheat_prob = {"NO_MOVES": 0.6, "AVAIL_CLAIMS": 0.1}
        # probability to decide to call cheat
        self.call_cheat_prob = {1: 0.06, 2: 0.011, 3: 0.28, 4: 0.47}
        # did I cheat in my last move
        self.is_last_cheat = False
        # cards that the opponent know I had
        self._my_known_cards = []

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

        # update known information
        if last_action == ActionEnum.CALL_CHEAT:
            self._table_known_cards = []
            self._opponent_claims = []
            if not self.is_last_cheat:
                for card in cards_revealed:
                    self._opponent_cards.append(card)
            else:
                # if I had cheat and the opponent called cheat he now know the cards I get
                for card in cards_revealed:
                    self._my_known_cards.append(card)

        if last_action == ActionEnum.MAKE_CLAIM:
            self._opponent_claims.append(last_claim)

        # reset variable
        self.is_last_cheat = False

        if last_claim:
            if self.need_to_call_cheat(opponent_count, last_action, last_claim):
                for move in honest_moves:
                    if isinstance(move, Call_Cheat):
                        return move

        move = self.get_best_move(last_claim, honest_moves)

        if isinstance(move, Take_Card) or isinstance(move, Call_Cheat) or isinstance(move, Claim):
            if isinstance(move, Claim):
                card_count = 0

                # adding to table known cards the cards that I really
                for card in self.cards:
                    if card.rank == move.rank and (card not in self._table_known_cards) and card_count < move.count:
                        self._table_known_cards.append(card)
                        card_count += 1
            return move

        self.is_last_cheat = True

        cheat_move = self.get_cheat(table_count)

        for card in cheat_move.cards:
            self._table_known_cards.append(card)

        return cheat_move

    def get_cheat(self, table_count):
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

        # decaying function of number of cards on the table - cheat less when risk is large
        r = 0.5 * exp(-0.1 * table_count)

        # choose cheat count
        while cheat_count < 4 and random.random() < r and len(self.cards) >= (cheat_count + 1):
            cheat_count += 1

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
            elif isinstance(move, Call_Cheat):
                scores[move] = self.call_cheat_prob[last_claim.count]
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

    def need_to_call_cheat(self, opponent_count, last_action, last_claim):
        if opponent_count == 0:
            return True

        rank_table_cards = 0
        if last_action == ActionEnum.MAKE_CLAIM:
            for card in self._table_known_cards:
                if card.rank == last_claim.rank:
                    rank_table_cards += 1
            for card in self.cards:
                if card.rank == last_claim.rank:
                    rank_table_cards += 1

            # there is only 4 cards of each number,
            # so I know for sure that the opponent had cheat
            if rank_table_cards + last_claim.count > 4:
                return True

        rank_claim_cards = 0
        for card in self._opponent_claims:
            if card.rank == last_claim.rank:
                rank_claim_cards += 1


        #if the total cards I know + the cards he claimed in the past + the cards he claimed jest now is more then 4
        #he either cheat now or he had cheat before
        #I would like to "claim cheat" in some probability depends on the size of the
        # TODO change the probabilities and add more conditions
        x = last_claim.count
        if rank_claim_cards + rank_table_cards + last_claim.count > 7:
            if random.random() > 0.8:
                return True
        if rank_claim_cards + rank_table_cards + last_claim.count > 6:
            if random.random() > 0.8:
                return True
        if rank_claim_cards + rank_table_cards + last_claim.count > 5:
            if random.random() > 0.8:
                return True
        if rank_claim_cards + rank_table_cards + last_claim.count > 4:
            if random.random() > 0.3:
                return True

        return False
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
cheat = Game(Agent_70("Demo 1"), Human("me"))
cheat.play()