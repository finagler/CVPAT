# Countdown Video Poker Analysis Toolkit (CVPAT)

The Countdown Video Poker Analysis Toolkit (this repository) is a
Python library for analyzing hands in Countdown Video Poker, and more
generally for writing agents to play the game. 

# What is Countdown Video Poker?

Countdown Video Poker is an experimental variant of [video
poker](https://en.wikipedia.org/wiki/Video_poker) that was developed by
**Strong Suit Games**. Like other video poker games the player is delt
a hand of five cards, discards up to all five cards and replaces them
from a supply, and receives a payoff for their final hand based on a
payout table. It differs in two ways:

1. Multiple decks are used (typically three), and five-of-a-kind
is added to the payout table.

1. Cards are drawn from a [shoe](https://en.wikipedia.org/wiki/Shoe_(cards)),
and unlike other video poker games they are not returned after a hand.
This means as play progresses the
player has more information about what cards remain in the shoe, and
can improve their expected payout based on that information. The game
includes a UI showing the relative frequency of each rank and suit
remaining in the shoe, so manual card-counting is not necessary.

The default version uses a three-deck shoe,
with play continuing until the shoe is depleted (with possibly fewer than five cards
available for replacement on the last hand). There is no ante, and the payout
table is as follows:

| hand             | payout |
|------------------|--------|
| royal flush      | 10000  |
| straight flush   | 1000   |
| five kind        | 200    |
| four kind        | 80     |
| full house       | 70     |
| flush            | 70     |
| straight         | 50     |
| three kind       | 20     |
| two pair         | 15     |
| jacks or better  | 10     |
| non-winning hand | 0      |

## Variations

There are a few main variations on the basic rules:

1. **Decks per shoe:** Size of the shoe, typically 2-4 decks.

1. **Payout Table:** How much each hand pays out.

1. **Hand-limit:** If a hand-limit is specified then play stops after
a set number of hands are played or the shoe is depleted (defined as
having nine or fewer cards left before drawing a new hand), whichever
comes first. Each hand uses between five and ten cards, so a hand
limit of up to 20 hands will always reach the hand-limit first in a
four-deck shoe, a hand limit of 21 to 39 could either hit the hand
limit or deplete the shoe depending on play, and a hand limit 40 and
above is effectively unlimited.

1. **Ante:** If an ante is specified then the player pays (bets) some
set amount per hand, and the goal is to maximize winnings or at least
get above zero. Note that ante is not the same as charging for an
entire shoe, since the latter does not depend on the number of hands
played. Ante can also be incorporated directly into the payout table
by subtracting a constant from each possible payout, with
"non-winning-hand" paying either zero or negative depending on the
mode.

1. **Allow short-shoe:** A hand will still be delt when the shoe has
between five and nine cards remaining, but the player can only discard
up to the number of cards remaining.

## What would an optimal strategy look like?

Standard video poker is relatively easy to analyze. For every
five-card hand there are 32 possible ways to discard cards, and for
each of those possible actions you can compute the probability of
drawing a winning hand from the rest of the deck. Multiplying those
probabilities by the payout table leads to an expected payout for each
possible action. Since each hand is independent (cards are shuffled
back in every hand), taking the action that maximizes the expected
payout of an individual hand will also maximize the expected payout of
a game.

It's more complex to compute the expected payout in Countdown Poker
because the probability of a winning hand depends on the contents of
the rest of the shoe, but it's still fairly straightforward and can be
computed exactly. What makes Countdown Poker tricky to analyze is that
hands are no longer independent, because the number of cards you
decide to hold affects how many cards will be in the shoe (and thus
how much you know about its contents) in subsequent hands. That means
any optimal strategy will need to trade off maximizing the expected
payout of the current hand with how that action changes subsequent
odds.

A good strategy should also be able to take downside and/or upside
risk into account. For example, a video poker player with a small
bankroll and a few hours to kill might be willing to take a slightly
lower expected payout from a machine if they're less likely to lose
their entire bankroll in the first few minutes. That's trading off
expected payout for lower downside risk. On the flip-side, a
high-roller who just wants to maximize their chance at a jackpot might
trade off expected payout for higher upside risk.

## Why this toolkit

When Countdown Video Poker was in development this toolkit was used to
help determine the payout table and other game parameters, and it was also
used to determine target scores for the daily puzzles that were
published.

The toolkit is also useful for analyzing some other Video Poker variants. In
particular, it's designed to:

* Compute the exact probability and expected payout of a given poker hand.
* Handle incomplete or multiple decks.
* Take into account additional advantage from holding or discarding extra cards.  

# Code layout

The code is organized as follows:

* [agent](./agent/): submodules for agents, and for estimating the
  advantage of holding additional cards for a particular rule-set.

* [data](./data/): pre-computed values used by agents (e.g. advantages
of holding cards for each position in a shoe).

* [game](./game/): submodules for implementing the game itself

* [notebooks](./notebooks/): Jupyter notebooks with example usage,
  analysis and other tools.

* [scripts](./scripts/): scripts for doing analysis and running the
  game from the command line.

* [stats](./stats/): submodules for computing statistics about the
  game, such as the expected payout of a hand.

