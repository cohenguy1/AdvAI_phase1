from __future__ import print_function, division

from enum import Enum
from itertools import combinations
from random import gauss  # doc sais it's faster than normalvariate, and it behaves the same

from cheat_game_client import Agent
from cheat_game_server import Card, Suit, Rank, ActionEnum, Claim, Take_Card, Call_Cheat, Cheat


def product(lst):
    result = 1
    for number in lst:
        result *= number
    return result


def binomial(n, k):
    """Number of ways to sample k elements from a set of n distinct elements"""

    if k < n - k:
        #  it's more efficient to have a larger k
        k = n - k

    nominator = product(xrange(k + 1, n + 1))  # == n! / k!
    denominator = product(xrange(1, (n - k) + 1))  # == (n - k)!
    return nominator // denominator  # == n! / (k! * (n - k)!)


TOTAL_NUMBER_OF_CARDS = len(Rank) * len(Suit)
NUMBER_OF_STARTING_CARDS = 8
PROBABILITY_TO_HAVE_SINGLE_CARD = (binomial(TOTAL_NUMBER_OF_CARDS - 1, NUMBER_OF_STARTING_CARDS - 1) /
                                   binomial(TOTAL_NUMBER_OF_CARDS, NUMBER_OF_STARTING_CARDS))


def average(lst):
    count = total = 0
    for item in lst:
        total += item
        count += 1

    if count:
        return total / count
    else:
        return 0


def standard_deviation(lst):
    avg = average(lst)
    return (average((item - avg) ** 2 for item in lst)) ** 0.5


def make_sure_values_sum_to_1(probabilities_dict):
    total_probability = sum(probabilities_dict.values())
    for key, value in probabilities_dict.items():
        probabilities_dict[key] = value / total_probability
    return probabilities_dict


def flatten(lst):
    try:
        for subgenerator in lst:
            for item in flatten(subgenerator):
                yield item
    except TypeError:
        yield lst


class CardStatus(Enum):
    NEVER_SEEN_IT = 'never seen this card'
    I_HAVE_IT = 'I have this card'
    ON_THE_TABLE = 'I know this card is on the table'
    OPPONENT_MAY_HAVE_IT = 'It is possible the opponent has it'


class CardPerception(object):
    def __init__(self, rank, suit):
        assert isinstance(rank, Rank)
        assert isinstance(suit, Suit)
        self.rank = rank
        self.suit = suit

        self.status = CardStatus.NEVER_SEEN_IT
        self.probability_opponent_has_it = PROBABILITY_TO_HAVE_SINGLE_CARD

    def __repr__(self):
        result = '%s of %s - %s' % (self.rank.value, self.suit.name, self.status.value)
        if 0 < self.probability_opponent_has_it:
            result = '%s, %.1f%% opponent has it' % (result, 100 * self.probability_opponent_has_it)
        return result

    def i_have_it(self):
        """I received the card"""
        self.status = CardStatus.I_HAVE_IT
        self.probability_opponent_has_it = 0

    def its_on_table(self):
        """I know for sure the card is on the table"""
        self.status = CardStatus.ON_THE_TABLE
        self.probability_opponent_has_it = 0

    def opponent_took_it(self):
        """I know the opponent took the card"""
        self.status = CardStatus.OPPONENT_MAY_HAVE_IT
        self.probability_opponent_has_it = 1

    def opponent_may_have_taken_it(self, probability_opponent_took_it):
        probability_opponent_didnt_have_it = 1 - self.probability_opponent_has_it
        probability_opponent_didnt_take_it = 1 - probability_opponent_took_it
        probability_opponent_doesnt_have_it = probability_opponent_didnt_have_it * probability_opponent_didnt_take_it
        self.probability_opponent_has_it = 1 - probability_opponent_doesnt_have_it

    def opponent_may_have_put_it_on_table(self, probability_opponent_put_it_on_table):
        probability_opponent_didnt_put_it = 1 - probability_opponent_put_it_on_table
        self.probability_opponent_has_it = self.probability_opponent_has_it * probability_opponent_didnt_put_it


class Perception(object):
    def __init__(self, parent):
        self.parent = parent
        self._cards = {rank: {suit: CardPerception(rank, suit)
                              for suit in Suit}
                       for rank in Rank}

    def __repr__(self):
        return str([card_perception for card_perception in self
                    if card_perception.status != CardStatus.NEVER_SEEN_IT])

    def __getitem__(self, args):
        if isinstance(args, Card):
            return self._cards[args.rank][args.suit]
        rank, suit = args
        assert isinstance(rank, Rank)
        assert isinstance(suit, Suit)
        return self._cards[rank][suit]

    def __iter__(self):
        for rank in Rank:
            for suit in Suit:
                yield self[rank, suit]

    def pprint(self):
        for card_perception in self:
            if card_perception.status != CardStatus.NEVER_SEEN_IT:
                print(card_perception)

    def cards_placed_on_table(self, *cards):
        for card in flatten(cards):
            self[card].its_on_table()

    def i_have_cards(self, *cards):
        for card in flatten(cards):
            self[card].i_have_it()

    def opponent_has_cards(self, *cards):
        for card in flatten(cards):
            self[card].opponent_took_it()

    def opponent_took_card_from_deck(self):
        probabilities_dict = {card: 1 - card.probability_opponent_has_it
                              for card in self
                              if card.status == CardStatus.NEVER_SEEN_IT}
        make_sure_values_sum_to_1(probabilities_dict)
        for card, probability_its_the_one_opponent_took in probabilities_dict.items():
            card.opponent_may_have_taken_it(probability_its_the_one_opponent_took)

    def opponent_made_claim_did_they_cheat(self, claim):
        opponent_matching_cards = [card for card in self._cards[claim.rank].values()
                                   if 0 < card.probability_opponent_has_it]

        if len(opponent_matching_cards) < claim.count:
            return 1

        probability_opponent_doesnt_have_enough = 1
        for combination in combinations(opponent_matching_cards, claim.count):
            probability_opponent_has_them = product(card.probability_opponent_has_it
                                                    for card in combination)
            probability_opponent_doesnt_have_enough *= 1 - probability_opponent_has_them

        total_combinations = binomial(len(opponent_matching_cards), claim.count)
        combinations_that_include_any_specific_card = binomial(len(opponent_matching_cards) - 1, claim.count - 1)
        probability_opponent_put_any_specific_card = ((1 - probability_opponent_doesnt_have_enough)
                                                      * combinations_that_include_any_specific_card / total_combinations)

        for card in opponent_matching_cards:
            card.opponent_may_have_put_it_on_table(probability_opponent_put_any_specific_card)

        # Be naive - assume they don't cheat if they can make an honest claim
        return probability_opponent_doesnt_have_enough

    def probability_opponent_thinks_i_cheat(self, rank, count, my_cards_count):
        # Be a a bit naive - assume they only think I cheat if they hold enough
        # cards to prove It. but not too naive - the less cards I have the more
        # likely the opponent will think I cheat

        my_matching_cards = [card for card in self._cards[rank].values()
                             if card.probability_opponent_has_it < 1]

        probability_opponent_cant_prove_i_cheat = 1
        num_cards_opponent_need_to_block = 5 - count
        for combination in combinations(my_matching_cards, num_cards_opponent_need_to_block):
            probability_opponent_has_them = product(card.probability_opponent_has_it
                                                    for card in combination)
            probability_opponent_cant_prove_i_cheat *= 1 - probability_opponent_has_them

        # The less cards I have, the more likely the opponent will think I cheat
        probability_opponent_cant_prove_i_cheat **= 2 - 1 / my_cards_count

        return 1 - probability_opponent_cant_prove_i_cheat

    def opponent_score_if_game_moves_to_rank(self, rank):
        score = 0
        for neighbor_rank in [rank.below(), rank.above()]:
            expected_num_cards = sum(card.probability_opponent_has_it for card in self if card.rank == neighbor_rank)
            # Give move weight to the ability to put more cards on the table.
            # Naively assume the probablity to be able to make all these moves
            # is the same.
            score += expected_num_cards ** 1.5
        return score


class Factors:
    # Values chosen by running a simple local maximum search
    rank_value_for_my_score = 1.135904779224186
    rank_value_for_opponent_score = 1.1034874473688419
    take_card_score = 1.2619948378823713
    make_claim_score = 0.7292234145625534
    call_cheat_score = 1.0334018835030054
    cheat_score = 1.653177647575028
    randomness = 0.899322884278146
    previously_called_cheat_penalty = 0.43743260560014624
    previously_cheated_penalty = 1.213817148615399


class Agent_25(Agent):
    def __init__(self, name):
        super(Agent_25, self).__init__(name)
        self.perception = Perception(self)
        self._first_round = True
        self.factors = Factors()
        self.times_previously_called_cheat = 0
        self.times_previously_cheated = 0

    def set_factors(self, **factors):
        for k, v in factors.items():
            assert not k.startswith('_')
            assert hasattr(self.factors, k)
            setattr(self.factors, k, v)

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

        if self._first_round:
            self._first_round = False
            self.perception.cards_placed_on_table(self.game.initial_card)

        if cards_revealed:
            # Assume it's the opponent that took the cards that were revealed.
            # If it was I that took them, they will appear in my deck, and so
            # when I call `i_have_cards()` they will be marked as mine.
            self.perception.opponent_has_cards(cards_revealed)

        self.perception.i_have_cards(self.cards)

        probability_opponent_cheat = 0
        if ActionEnum.TAKE_CARD == last_action:
            self.perception.opponent_took_card_from_deck()
        elif ActionEnum.MAKE_CLAIM == last_action:
            probability_opponent_cheat = self.perception.opponent_made_claim_did_they_cheat(last_claim)
        elif ActionEnum.CALL_CHEAT == last_action:
            pass

        scores_for_ranks = {}
        for rank in [self.table.top_rank().below(),
                     self.table.top_rank(),
                     self.table.top_rank().above()]:
            scores_for_ranks[rank] = self.score_if_game_moves_to_rank(rank)
        moves_scores = {}
        for move in honest_moves:
            new_rank = self.table.top_rank()
            if isinstance(move, Claim):
                score = self.score_make_claim_move(move,
                                                   table_count=table_count,
                                                   opponent_count=opponent_count)
                new_rank = move.rank
            elif isinstance(move, Take_Card):
                score = self.score_take_card_move(opponent_count=opponent_count)
            elif isinstance(move, Call_Cheat):
                score = self.score_call_cheat_move(probability_opponent_cheat=probability_opponent_cheat,
                                                   table_count=table_count,
                                                   opponent_count=opponent_count)

            score_for_new_rank = scores_for_ranks[new_rank]
            score += score_for_new_rank

            moves_scores[move] = score

        for cheat_rank in [self.table.top_rank().below(),
                           self.table.top_rank().above()]:
            for cheat_cards in self.generate_cheat_moves_cards(cheat_rank):
                cheat_move = Cheat(cheat_cards, cheat_rank, len(cheat_cards))
                score = self.score_cheat_move(cards=cheat_cards,
                                              rank=cheat_rank,
                                              table_count=table_count,
                                              opponent_count=opponent_count)
                score += scores_for_ranks[cheat_rank]
                moves_scores[cheat_move] = score

        scores_standard_deviation = standard_deviation(moves_scores.values())
        magnitude_of_randomness = scores_standard_deviation * self.factors.randomness
        for move, score in moves_scores.items():
            moves_scores[move] = score + gauss(0, magnitude_of_randomness)

        best_move = max(moves_scores.keys(), key=moves_scores.__getitem__)

        if isinstance(best_move, Cheat):
            self.times_previously_cheated += 1 + self.times_previously_cheated

        if isinstance(best_move, Call_Cheat):
            self.times_previously_called_cheat += 1 + self.times_previously_called_cheat

        return best_move

    def score_if_game_moves_to_rank(self, rank):
        my_score = 0
        for neighbor_rank in [rank.below().below(),  # opponent claims below - can I claim below?
                              rank.below(),  # opponent doesn't claim - can I claim below?
                              rank,  # opponent claim - can I claim in the opposite direction?
                              rank.above(),  # opponent doesn't claim - can I claim above?
                              rank.above().above(),  # opponent claims above - can I claim above?
                              ]:
            num_cards = sum(1 for card in self.cards if card.rank == neighbor_rank)
            # Give move weight to the ability to put more cards on the table.
            # Naively assume the probablity to be able to make all these moves
            # is the same.
            my_score += num_cards ** 1.5
        my_score *= self.factors.rank_value_for_my_score
        opponent_score = self.perception.opponent_score_if_game_moves_to_rank(rank)
        opponent_score *= self.factors.rank_value_for_opponent_score
        return my_score - opponent_score

    def score_make_claim_move(self, make_claim_move, table_count, opponent_count):
        if make_claim_move.count == len(self.cards):
            # This move is guaranteed to win the game - of course we take it!
            # (also, if we go on we'll get a division by zero when calculating
            # `value_for_me`...)
            return float('inf')
        probability_opponent_call_cheat = self.perception.probability_opponent_thinks_i_cheat(rank=make_claim_move.rank,
                                                                                              count=make_claim_move.count,
                                                                                              my_cards_count=len(self.cards))
        value_if_opponent_calls_cheat = (opponent_count + table_count) / (len(self.cards) - make_claim_move.count)
        value_if_opponent_doesnt_call_cheat = opponent_count / (len(self.cards) - make_claim_move.count)
        expected_value = (probability_opponent_call_cheat * value_if_opponent_calls_cheat
                          + (1 - probability_opponent_call_cheat) * value_if_opponent_doesnt_call_cheat)
        return self.factors.make_claim_score * expected_value

    def score_take_card_move(self, opponent_count):
        value_if_i_take_a_card = opponent_count / len(self.cards)
        return self.factors.take_card_score * value_if_i_take_a_card

    def score_call_cheat_move(self, probability_opponent_cheat, table_count, opponent_count):
        value_if_it_is_a_cheat = (opponent_count + table_count) / len(self.cards)
        value_if_it_is_not_a_cheat = opponent_count / (len(self.cards) + table_count)
        value_if_i_call_cheat = (probability_opponent_cheat * value_if_it_is_a_cheat
                                 + (1 - probability_opponent_cheat) * value_if_it_is_not_a_cheat)
        value_if_it_is_not_a_cheat *= self.factors.call_cheat_score
        return value_if_i_call_cheat - 10 * self.factors.previously_called_cheat_penalty * self.times_previously_called_cheat

    def score_cheat_move(self, cards, rank, table_count, opponent_count):
        if len(cards) == len(self.cards):
            # Never finish the game with a cheat - the opponent's best response
            # will always be to call cheat, because they have nothing to lose
            # if they call wrong.
            # (also, if we go on we'll get a division by zero when calculating
            # `value_for_me`...)
            return float('-inf')

        probability_opponent_call_cheat = self.perception.probability_opponent_thinks_i_cheat(rank=rank,
                                                                                              count=len(cards),
                                                                                              my_cards_count=len(self.cards))
        value_if_opponent_calls_cheat = opponent_count / (len(self.cards) + table_count)
        value_if_opponent_doesnt_call_cheat = opponent_count / (len(self.cards) - len(cards))
        expected_value = (probability_opponent_call_cheat * value_if_opponent_calls_cheat
                          + (1 - probability_opponent_call_cheat) * value_if_opponent_doesnt_call_cheat)
        return self.factors.cheat_score * expected_value - \
            10 * self.factors.previously_cheated_penalty * self.times_previously_cheated

    def generate_cheat_moves_cards(self, target_rank):
        # No point in half-cheating - not when I don't take into account the
        # ranks of cards I will have after the move
        cards = [card for card in self.cards if card.rank != target_rank]
        for count in range(1, min(len(cards) + 1, 4)):
            yield cards[:count]

