from enum import Enum, IntEnum
import random
from abc import ABCMeta, abstractmethod
import pickle


class Suit(IntEnum):
    """
    Enumeration of the four suits in the deck of cards\n
    Used by class Card
    """
    HEART = 1
    DIAMOND = 2
    CLUB = 3
    SPADE = 4


class Rank(IntEnum):
    """
    Enumeration of the 13 ranks of cards in a standard deck of cards\n
    Used by class Card\n
    helper functions include:\n
    above(), below() - return appropriate adjacent rank\n
    dist(other) - numerical "distance" between two ranks
    """
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13

    def above(self):
        """
        :return: a rank higher than current rank. for King return Ace.
        """
        r = int(self)
        r += 1
        if r > int(Rank.KING): r = int(Rank.ACE)
        return Rank(r)

    def below(self):
        """
        :return: a rank lower than current rank. for Ace return King.
        """
        r = int(self)
        r -= 1
        if r < int(Rank.ACE): r = int(Rank.KING)
        return Rank(r)

    def dist(self, other):
        """
        :param other: Rank object
        :return: ordinal difference between ranks using cyclic boundaries, i.e disk(KING,ACE)=1
        """
        r1 = int(self)
        r2 = int(other)
        d = r1 - r2
        if d < 0: d = -d
        if d > 6: d = 13 - d
        return d

    def __str__(self):
        return Card.rank2str[int(self._value_)]


class Card(object):
    """
    a playing card with a rank and suit (see enumeration definition)\n
    methods:\n
    * index() - return ordinal \in 0..51
    * adjacent(other) - returns True if other card's rank is adjacent to self
    """

    def __init__(self, rank, suit):
        """
        :param rank: card rand of enumeration type Rank
        :param suit: card suit of enumeration type Suit
        :return: new Card object
        """
        self.rank, self.suit = rank, suit

    def adjacent(self, other):
        """
        :param other: Card object
        :return: True iff either card is joker or rank.dist(other.rank)=1, i.e. cards are of adjecent rank
        """
        if isinstance(self, Joker) or isinstance(other, Joker): return True
        if (((self.rank == Rank.ACE) and (other.rank == Rank.KING)) or
                ((other.rank == Rank.ACE) and (self.rank == Rank.KING))):
            return True
        d = int(self.rank) - int(other.rank)
        if d == 1 or d == -1:
            return True
        return False

    def index(self):
        """
        :return: hashes rank and suit to ordinal \in 0..51 for easy indexing in Deck
        """
        return (int(self.rank) - 1) * 4 + int(self.suit) - 1

    def __cmp__(self, other):
        return cmp((int(self.rank), int(self.suit)), (int(other.rank), int(other.suit)))

    rank2str = [None, 'A ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ', '10', 'J ', 'Q ', 'K ']

    def __str__(self):
        if isinstance(self, Joker):
            return '* '
        return Card.rank2str[int(self.rank)]


class Joker(Card):
    """
    Joker card can be added to deck\n
    card.adjacent(Joker) V Joker.adjacent(card) is always True
    """

    def __init__(self):
        pass


class Deck:
    """
    A standard deck of 52 cards, optionally with two Jokers
    """

    def __init__(self, jokers=False):
        """
        :param jokers: optional flag. if True, two Jokers are added to Deck
        :return: new Deck object with 52 cards (and optionally two jokers)
        """
        self._cards = [Card(rank, suit) for rank in Rank for suit in Suit]
        if jokers: self._cards.extend([Joker(), Joker()])
        random.shuffle(self._cards)

    def deal_top(self):
        """
        removes top card from deck\n
        :return: top card from deck
        """
        size = len(self._cards)
        assert size > 0, "can't take cards from an empty deck"
        return self._cards.pop()

    def put_back(self, card):
        """
        add a card to the deck\n
        :param card: card to add
        :return: None
        """
        if isinstance(card, Joker):
            assert len(
                [1 for c in self._cards if isinstance(c, Joker)]) < 2, "Can't put back a Joker to a deck with 2 Jokers"
        else:
            assert all([card != c for c in self._cards]), "can't put back a card which is in the deck"
        self._cards.insert(random.randrange(len(self._cards)), card)

    def deal_specific(self, rank):
        available_cards = [card for card in self._cards if card.rank == rank]
        assert len(available_cards) > 0, "specified rank not in deck"
        card = random.choice(available_cards)
        self._cards.remove(card)
        return card

    def empty(self):
        """
        :return: True iff Deck is empty
        """
        return not self._cards


class Player(object):
    """
    Base class for cheat game players (human and agents)\n
    includes common methods and attributes
    """

    def __init__(self, name, game=None, id=0):
        """
        Base initializor called hierarchically from descendants\n
        :param name: name of player
        :param game: optional Game object
        :param id: optional id (currently overriden by game object)
        :return: Player object
        """
        self.name = name
        self.cards = []
        self.connect_to_game(game)
        self.id = id

    def get_id(self):
        """
        :return: Player.id
        """
        return self.id

    def set_id(self, id):
        """
        :param id: new id of Player object
        :return:  None
        """
        self.id = id

    def connect_to_game(self, game):
        """
        connect Player to Game object, i.e. store pointers to game/table/deck objects\n
        :param game: Game object
        :return: None
        """
        self.game = game
        if self.game:
            self.table = self.game.table
            self.deck = self.game.deck

    def take_card_from_deck(self, silent=False):
        """
        takes the top card from the deck\n
        :return: None
        """
        self.cards.append(self.deck.deal_top())

    def take_cards_from_table(self):
        """
        Take all the cards from the table\n
        Called by loser of "Call Cheat" event\n
        :return: cards
        """
        cards = self.table.take_cards()
        print 'Player {0} "{1}" received the cards from the table'.format(self.id, self.name)
        print '  Cards: ', ' '.join([str(card).strip() for card in cards])
        self.cards.extend(cards)
        return cards

    def hand_size(self):
        """
        :return: number of player's cards
        """
        return len(self.cards)

    def sort_cards(self):
        """
        sort in-place player's cards by rank\n
        :return: None
        """
        self.cards.sort(key=lambda card: int(card.rank))

    def remove_card_by_rank(self, rank, count):
        """
        Used to generate a legal claim, removing 'count' cards of 'rank' from player's hand\n
        :param rank: rank of cards to remove
        :param count: number of cards to remove
        :return: removed cards
        """
        cards = []
        for card in self.cards:
            if count == 0: break
            if card.rank == rank:
                cards.append(card)
                self.cards.remove(card)
        assert count == 0, \
            'remove_card_by rank: trying to remove ' + count + ' cards of rank: ' + rank + \
            '. but ' + count + ' cards not available'
        return cards

    def remove_random_cards(self, count):
        """
        remove 'count' cards randomly\n
        might be useful when playing a 'cheat' claim\n
        :param count: number of cards
        :return: the cards removed
        """
        cards = []
        assert self.hand_size() >= count, \
            'remove_random_cards: trying to remove ' + count + 'cards from ' + self.hand_size() + ' cards in hand.'
        for _ in xrange(count):
            card = self.cards.pop(random.randrange(self.hand_size()))
            cards.append(card)
        return cards

    def possible_honest_moves(self):
        """
        :return: a list of legal truthful moves
        """
        top_card_rank = self.table.top_rank()
        rank_above = Rank.above(top_card_rank)
        rank_below = Rank.below(top_card_rank)
        above_cards = [card for card in self.cards if isinstance(card, Joker) or card.rank == rank_above]
        below_cards = [card for card in self.cards if isinstance(card, Joker) or card.rank == rank_below]
        moves = []
        if not self.deck.empty():
            moves.append(Take_Card())
        if self.table.claims and self.game.get_state()['LAST_ACTION'] == ActionEnum.MAKE_CLAIM:
            moves.append(Call_Cheat())
        if above_cards:
            for claim_size in xrange(1, len(above_cards) + 1):
                moves.append(Claim(rank_above, claim_size))
        if below_cards:
            for claim_size in xrange(1, len(below_cards) + 1):
                moves.append(Claim(rank_below, claim_size))
        return moves

    def honest_claim_with_most_cards(self):
        """
        :return: a legal claim with the most cards or None
        """
        moves = self.possible_honest_moves()
        claim = None
        count = 0
        for move in moves:
            if isinstance(move, Claim):
                if move.count > count:
                    claim, count = move, move.count
        if count == 0:
            return None
        return claim

    def make_honest_claim(self, claim):
        """
        perform an honest claim:\n
        * remove cards from hand (based on claim's rank)
        * add cards to table
        * add claim to claims list
        :param claim: containing the rank and count of cards
        :return: None
        """
        cards = [card for card in self.cards if card.rank == claim.rank]
        cards = cards[:claim.count]
        self.make_claim(cards, claim)

    def make_claim(self, cards, claim):
        """
        generalized claim making, supporting untruthful (cheat) claims\n
        :param cards: cards to be removed and placed on table
        :param claim: containing the rank and count of cards
        :return: None
        """
        assert claim.count == len(cards), "number of cards played must match the claim"
        assert all(card in self.cards for card in cards), "must play with cards in 'hand'"
        self.table.make_claim(claim)
        self.table.add_cards(cards)
        for card in cards:
            self.cards.remove(card)

    def call_cheat(self):
        """
        perform the call cheat action\n
        Game object deduces whether last claim was truthful or not and adds table cards to the appropriate player\n
        :return: None
        """
        self.game.call_cheat(self.id)

    def make_move(self):
        pass

    def won(self):
        """
        callback function. called by Game object when player has won\n
        :return: None
        """
        print "Player {0:1d} - \"{1}\": hurray I won!".format(self.id, self.name)

    def lost(self):
        """
        callback function. called by Game object when player has lost\n
        :return: None
        """
        print "Player {0:1d} - \"{1}\": boohoo I lost!".format(self.id, self.name)


class Action():
    """
    Abstract base class for possible game action classes
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """
        abstract initializor
        """
        pass


class Claim(Action):
    """
    Claim action class
    """

    def __init__(self, rank, count):
        """
        :param rank: card rank
        :param count:  number of cards
        :return: initialized Claim object
        """
        assert isinstance(rank, Rank), 'rank is required to be of the enumeration Rank type'
        assert count > 0 and count <= 4, 'number of cards have to be between 1 and 4'
        self.rank = rank
        self.count = count


class Call_Cheat(Action):
    """
    Call Cheat action
    """

    def __init__(self):
        pass


class Take_Card(Action):
    """
    Take Card from deck action
    """

    def __init__(self):
        pass


class Cheat(Action):
    """
    Cheat (untruthful claim) action
    """

    def __init__(self, cards=None, rank=None, count=None):
        """
        :param cards: list of cards from players hand
        :param rank: card rank
        :param count:  number of cards
        :return: initialized Cheat object
        """
        self.cards = cards
        self.rank = rank
        self.count = count


class Human(Player):
    """
    Human class derived from Player class\n
    enables console interaction for making moves
    """

    def __init__(self, name='Human'):
        super(Human, self).__init__(name)

    def make_move(self):
        """
        override Player.make_move() called by Game object\n
        displays current player's cards and possible moves\n
        allows human player to select a move\n
        :return: None
        """
        state = self.game.get_state()
        number_of_opponent_cards = state[3 - self.id]
        print
        print "Player {0:1d} ({1:s}) turn".format(self.id, self.name)
        print "================" + "=" * len(self.name)
        last_claim = self.table.last_claim()
        print "Last Claim: {0} cards of rank {1}".format(last_claim.count, str(last_claim.rank))
        print "Number of opponenr cards: {0:2d}".format(number_of_opponent_cards)
        print "Your Cards: ", ','.join([str(card) for card in self.cards])
        print "possible moves:"
        moves = self.possible_honest_moves()
        print "Available Moves: "
        for index, move in enumerate(moves):
            if isinstance(move, Take_Card):
                print "{0:2d} - Take card from deck".format(index + 1)
            elif isinstance(move, Call_Cheat):
                print "{0:2d} - Call \"Cheat\" to previous claim".format(index + 1)
            elif isinstance(move, Claim):
                print "{0:2d} - Place {1:1d} cards of rank {2:1s}".format(index + 1, move.count,
                                                                          Card.rank2str[move.rank])
        print "{0:2d} - to cheat".format(0)
        invalid = True
        while invalid:
            selection = 0
            try:
                selection = int(raw_input('Select Move: '))
            except ValueError:
                print 'non numeric selection'
                continue
            if selection >= 0 and selection <= len(moves):
                invalid = False
        if selection == 0:
            top_card_rank = self.table.top_rank()
            rank_above = Rank.above(top_card_rank)
            rank_below = Rank.below(top_card_rank)
            rank_selection = 0
            print "Possible ranks:"
            print "1 - ", rank_above
            print "2 - ", rank_below
            while rank_selection != 1 and rank_selection != 2:
                try:
                    rank_selection = int(raw_input('select rank for chear claim: '))
                except ValueError:
                    print "please enter '1' or '2'"
                    continue
            if rank_selection == 1:
                cheat_rank = rank_above
            else:
                cheat_rank = rank_below
            cheat_count = 0
            max_cheat_count = min(4, len(self.cards))
            while cheat_count < 1 or cheat_count > max_cheat_count:
                try:
                    cheat_count = int(raw_input('How many cards (1-{0:d}): '.format(max_cheat_count)))
                except ValueError:
                    print "please enter a number between '1' and '4'"
                    continue
            claim_cards = []
            for index in xrange(cheat_count):
                for i, c in enumerate(self.cards):
                    if c not in claim_cards:
                        print "{0:d}: [{1:s}] ".format(i + 1, str(c).strip()),
                print
                selected = False
                while not selected:
                    try:
                        card_index = int(raw_input('select card number (enter card index):'))
                    except ValueError:
                        print "please enter valid number of card"
                        continue
                    if card_index > 0 and card_index <= len(self.cards):
                        selected = True
                claim_cards.append(self.cards[card_index - 1])
            self.make_claim(claim_cards, Claim(cheat_rank, cheat_count))
        else:
            move = moves[selection - 1]
            if isinstance(move, Take_Card):
                self.take_card_from_deck()
            elif isinstance(move, Call_Cheat):
                self.call_cheat()
            elif isinstance(move, Claim):
                self.make_honest_claim(move)


class Table(object):
    """
    Table class to hold cards and claims
    """

    def __init__(self):
        """
        :return: Initialized Table object
        """
        self._cards = []
        self.claims = []
        self.initial_dummy_claim = None

    def add_cards(self, cards):
        """
        adds cards to table \n
        :param cards: cards to add
        :return: None
        """
        self._cards.extend(cards)
        if not self.initial_dummy_claim:
            assert len(cards) == 1, "place only one card for initial move"
            self.initial_dummy_claim = Claim(cards[0].rank, 1)

    def make_claim(self, claim):
        """
        adds the claim to claim list\n
        :param claim: new claim containing rank and count of cards
        :return: None
        """
        self.claims.append(claim)

    def size(self):
        """
        :return: the number of cards on the table
        """
        return len(self._cards)

    def top_rank(self):
        """
        :return: the CLAIMED top rank on the table
        """
        return self.last_claim().rank

    def last_claim(self):
        """
        :return: the last claim (rank and count)
        """
        if not self.claims:
            return self.initial_dummy_claim
        return self.claims[-1]

    def _true_claim(self):
        last_claim = self.claims[-1]
        last_cards = self._cards[-last_claim.count:]
        return all(card.rank == last_claim.rank for card in last_cards)

    def take_cards(self):
        """
        "hands over" the cards on the table\n
         called by player after losing a "call cheat" event\n
        :return: the cards on the table
        """
        cards = list(self._cards)
        self._cards = []
        return cards


class ActionEnum(IntEnum):
    TAKE_CARD = 1
    CALL_CHEAT = 2
    MAKE_CLAIM = 3


class Game:
    """
    Game class - calls players' make_move alternatively and declares winner
    """

    def __init__(self, player1=Player(''), player2=Player(''), with_jokers=False, lose_on_30=False):
        """
        :param player1: first player  (derived from abstract class Player)
        :param player2: second player (derived from abstract class Player)
        :param with_jokers: optional flag. when true - add two jokers to deck
        :param lose_on_30:  optional flag. when true - game ends when a player has at least 30 cards
        :return: an initialized Game object
        """
        self.cheat_flag = 0
        self.cheat_called_on_prev_move = False
        self.no_new_claim = True
        random.seed()
        self.cur_player = -1
        self.with_jokers = with_jokers
        self.lose_on_30 = lose_on_30
        self.state = []
        self.deck = Deck(jokers=with_jokers)
        self.table = Table()
        player1.set_id(1);
        player1.connect_to_game(self)
        player2.set_id(2);
        player2.connect_to_game(self)
        self.__players = [player1, player2]
        self.winner = None
        self.loser = None
        self.last_action = ""
        self.cards_revealed = None
        # deal 2 8 card hands
        for player in self.__players:
            for _ in xrange(8):
                player.take_card_from_deck(silent=True)
        # place a card on the table (not a Joker)
        card = None
        while True:
            card = self.deck.deal_top()
            if not isinstance(card, Joker):
                break
            self.deck.put_back(card)
        self.initial_card = card
        self.table.add_cards([card])
        self.save_state()

    def get_initial_card(self):
        """
        :return: initial card - top card on table at the beginning of the game
        """
        return self.initial_card

    def save_state(self):
        """
        saves the current state of the game\n
        state contains both players' cards; cards on table; list of claims\n
        player's move can be deduced from successive game states\n
        :return: None
        """
        self.state.append([self.__players[0].cards, self.__players[1].cards, self.table._cards, self.table.claims])

    def get_state(self):
        """
        :return: count distribution of cards, i.e. number of cards for each player, table, deck, and last action
        """
        return {1: len(self.__players[0].cards), 2: len(self.__players[1].cards),
                'TABLE': len(self.table._cards), 'DECK': len(self.deck._cards),
                'LAST_ACTION': self.last_action, 'CARDS_REVEALED': self.cards_revealed}

    def end_of_game(self):
        """
        check for end of game and call winner/loser callback functions of players\n
        :return: True if end of game, False otherwise
        """
        done = False
        winner = 0;
        loser = 0
        for player in self.__players:
            # declare victory when player "got rid" of all his cards and his opponent had a chance to play
            if player.hand_size() == 0 and (player.get_id() - 1) == self.cur_player:
                done = True
                winner = player.get_id() - 1
                loser = 1 - winner
                break
            if self.lose_on_30:
                if player.hand_size() >= 30:
                    done = True
                    loser = player.get_id() - 1
                    winner = 1 - loser
                    break
        if done:
            self.winner = self.__players[winner]
            self.loser = self.__players[loser]
        return done

    def call_cheat(self, id):
        """
        called by player performing "call cheat" action\n
        determines whether last claim was truthful or not and give the cards on the table to the loser\n
        :param id: calling player's id (1 or 2)
        :return: None
        """
        assert id == 1 or id == 2, "invalid player id calling cheat"
        if self.table._true_claim():
            loser = self.__players[id - 1]
        else:
            loser = self.__players[2 - id]
        self.cards_revealed = loser.take_cards_from_table()
        self.cheat_called_on_prev_move = True
        self.cheat_flag = len(self.table.claims)

    def after_calling_cheat(self):
        return self.cheat_called_on_prev_move

    def update_cheat_flag(self):
        if self.cheat_called_on_prev_move:
            if len(self.table.claims) > self.cheat_flag:
                self.cheat_called_on_prev_move = False

    def last_claim(self):
        """
        :return: last claim
        """
        if not self.table.claims or self.no_new_claim:
            return None
        return self.table.last_claim()

    def play(self):
        """
        main game loop\n
        :return: None
        """
        # randomly choose first player
        self.cur_player = random.randrange(2)
        while not self.end_of_game():
            self.__players[self.cur_player].sort_cards()
            self.update_cheat_flag()
            prev_claim = None
            if self.table.claims:
                prev_claim = self.table.last_claim()
            deck_num_cards = len(self.deck._cards)
            table_num_claims = len(self.table.claims)
            self.__players[self.cur_player].make_move()
            if len(self.deck._cards) < deck_num_cards:
                self.last_action = ActionEnum.TAKE_CARD
            elif len(self.table.claims) > table_num_claims:
                self.last_action = ActionEnum.MAKE_CLAIM
            else:
                self.last_action = ActionEnum.CALL_CHEAT
            if self.last_action != ActionEnum.CALL_CHEAT:
                self.cards_revealed = None
            if self.table.claims:
                self.no_new_claim = (prev_claim == self.table.last_claim())
            self.save_state()
            self.cur_player = 1 - self.cur_player
        self.winner.won()
        self.loser.lost()

    def save_state_to_file(self, fname):
        """
        dump game state to file, using python's pickle\n
        :param fname: file name
        :return: None
        """
        pickle.dump(self.state, open(fname, "wb"))
