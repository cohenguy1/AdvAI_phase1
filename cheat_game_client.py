import random
from math import exp
from collections import defaultdict

from cheat_game_server import Game, ActionEnum
from cheat_game_server import Player, Human
from cheat_game_server import Claim, Take_Card, Cheat, Call_Cheat
from cheat_game_server import Rank, Suit, Card


class Agent(Player):
    def __init__(self, name):
        super(Agent, self).__init__(name)

    def make_claim(self, cards, claim):
        if not self._silent: print 'making claim: {0:1d} cards of rank {1}'.format(claim.count, str(claim.rank))
        super(Agent, self).make_claim(cards, claim)

    def make_honest_claim(self, claim):
        super(Agent, self).make_honest_claim(claim)

    def take_card_from_deck(self):
        if not self._silent: print 'Taking Card from deck'
        super(Agent, self).take_card_from_deck()

    def call_cheat(self):
        if not self._silent: print 'Calling "Cheat!"'
        super(Agent, self).call_cheat()
        state = self.game.get_state()

    def make_move(self):
        if not self._silent: print
        if not self._silent: print 'Player {0:1d} ({1:s}) turn'.format(self.id, self.name)
        if not self._silent: print "================"+"="*len(self.name)
        honest_moves = self.possible_honest_moves()
        state = self.game.get_state()
        opponent_count = state[3 - self.id]
        deck_count = state['DECK']
        table_count = state['TABLE']
        last_action = state['LAST_ACTION']
        cards_revealed = state['CARDS_REVEALED']
        last_claim = self.game.last_claim()

        if last_claim:
            if not self._silent: print "Last Claim: {0} cards of rank {1}".format(last_claim.count, str(last_claim.rank))
        if not self._silent: print "Number of opponent cards: {0:2d}".format(opponent_count)
        if not self._silent: print "Your Cards: ", ','.join([str(card) for card in self.cards])
        if not self._silent: print "Deck count: {0}".format(len(self.deck._cards))
        if not self._silent: print "Table count: {0}".format(len(self.table._cards))

        # if opponent placed his last cards on the table - call_cheat or lose
        action = self.agent_logic(deck_count, table_count, opponent_count,
                                  last_action, last_claim, honest_moves, cards_revealed)
        # print action
        if not action:
            for move in honest_moves:
                if isinstance(move, Call_Cheat):
                    action = move

        assert (action in honest_moves or isinstance(action, Cheat))
        if isinstance(action, Call_Cheat):
            self.call_cheat()
        elif isinstance(action, Claim):
            self.make_honest_claim(action)
        elif isinstance(action, Take_Card):
            self.take_card_from_deck()
        elif isinstance(action, Cheat):
            self.make_claim(action.cards, Claim(action.rank, action.count))


class DemoAgent(Agent):
    def __init__(self, name):
        super(DemoAgent, self).__init__(name)
        self.cheat_prob = {"NO_MOVES": 0.6, "AVAIL_CLAIMS": 0.1}
        self.call_cheat_prob = {1: 0.06, 2: 0.011, 3: 0.28, 4: 0.47}

    def agent_logic(self, deck_count, table_count, opponent_count,
                    last_action, last_claim, honest_moves, cards_revealed):
        """
        This function implements action logic / move selection for the agent\n
        :param deck_count:
        :param table_count:
        :param opponent_count:
        :param last_action: ActionEnum.TAKE_CARD or .MAKE_CLAIM or .CALL_CHEAT
        :param last_claim:
        :param honest_moves: a list of available actions, other than making a false ("cheat") claim
        :param cards_revealed: if last action was "call cheat" cards on table were revealed
        :return: Action object Call_Cheat or Claim or Take_Card or Cheat
        """
        scores = {}
        if opponent_count == 0:
            for move in honest_moves:
                if isinstance(move, Call_Cheat):
                    return move
        available_claim = False
        for move in honest_moves:
            if isinstance(move, Claim):
                scores[move] = move.count
                available_claim = True
            elif isinstance(move, Take_Card):
                scores[move] = 0.6
            elif isinstance(move, Call_Cheat):
                if last_claim:
                    scores[move] = self.call_cheat_prob[last_claim.count]
                else:
                    scores[move] = 0.0
        if available_claim:
            scores[Cheat()] = self.cheat_prob["AVAIL_CLAIMS"]
        else:
            scores[Cheat()] = self.cheat_prob["NO_MOVES"]
        # randomize scores add random \in [-0.5..0.5)
        for move, score in scores.iteritems():
            scores[move] = score + 0.5 * (2.0 * random.random() - 1)
        # select move based on max score
        move = max(scores, key=scores.get)
        if isinstance(move, Take_Card):
            return move
        elif isinstance(move, Call_Cheat):
            return move
        elif isinstance(move, Claim):
            return move
        elif isinstance(move, Cheat):
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
            while cheat_count < 4 and random.random() < r and len(self.cards) >= (cheat_count + 1):
                cheat_count += 1
            # select cards furthest from current claim rank
            dist = defaultdict(int)
            for ind, card in enumerate(self.cards):
                dist[card] = cheat_rank.dist(card.rank)
            claim_cards = sorted(dist, key=dist.get)[:cheat_count]
            return Cheat(claim_cards, cheat_rank, cheat_count)



