# 🍎 Apple Destroyer

## Game Explanation

There is a board of apples.  
You can make rectangular selections to clear apples.  
You can only pop the apples if the selected apples **sum to exactly 10**.  

Try the game here: [Fruit Box Game](https://en.gamesaien.com/game/fruit_box/)

---

## Synopsis

I found this game and became obsessed with it for a while.  
I cracked a score of 100 within a few days, and hit 109 after about a week.

After that, I just couldn't get a higher score.  
Sometimes I was too slow, and sometimes, I simply ran out of moves.

To confirm this, I took screenshots near the timeout and continued solving in MS Paint.  
Even with no time limit, I'd hit a point with no valid selections left.  
With unlimited time, a good run would get me to around **120**.

When I thought I was finally done with the game, I suffered severe insomnia.

One question lingered in my head, repeating over and over...

> **"Is this game solvable?"**  
> Can you completely clear the board?  
> What is the theoretical limit?  
> Imagine a world with no apples.

That marked the beginning of the **Apple Destroyer** endeavor.

---

## 🛠 Project

### 🔍 Attempt 1: Brute Force

My first approach was to try **all combinations of moves**, selecting the sequence that leads to the highest score.

But the search space was way too large:

- Around 20–40 possible moves per board state.
- This number stays fairly high due to cascading new move opportunities.
- Assuming each move clears ~2.33 apples (based on a 2:1 ratio of 2-apple to 3-apple moves):

```
170 apples / 2.33 ≈ 73 total moves
```
- Estimated search space:
```
(35 possible selections)^53 non-endgame moves + (20 endgame moves)! ≈ 6.85 × 10^81 states
```
That's **more board states than atoms in the universe**.

### Attempt 2: Monte Carlo Tree Search (MCTS)

I pivoted to using a **Monte Carlo Tree Search algorithm with heuristics**.

> MCTS is a smart way of navigating board states using a tree structure.  
> The magic lies in the **heuristic** — a formula to estimate how "good" a board state is.

#### Heuristic Design

A naive heuristic would be:
- **Total score so far** — but this increases with depth, so early branches dominate.

Instead, I thought about **bad moves**:

Example of a bad move:
```
[1 1 1 1 1 1 1 1 1 1]
```

Clearing this wastes valuable 1's, which are wild cards.
- 9's can *only* be cleared with 1's.
- So, using up all 1's makes it impossible to clear 9's later.

This applies similarly (but less severely) to 2's, 3's, and 4's.

#### Heuristic Attempts

1. **Mean board value** — worked okay.  
   Achieved a score of **135**.

2. But... this skews toward:
   - High values → early terminal boards → breadth-first behavior.
   - Low values (e.g. treating empty spaces as 0's) → depth-first behavior.

3. **# of valid moves** — also biased toward early game.

4. **# of valid moves + depth** — worked best!  
   Final result: **156**

Tried other heuristics, but none came close to 156.

---

## Conclusion

I still don’t know if the board is truly solvable, but I now know:

- You can get **really close** to clearing it.
- On a very lucky run, I suspect scores of **~160 to ~165** are possible.

---

## Ending Notes / Future Ideas

If I revisit this project, I'd like to explore:

1. **Parallelization with threads or GPU**  
   - Increase performance to allow deeper searches.

2. **Train a CNN as the heuristic**  
   - Similar to techniques used in **AlphaGo** and high-level chess engines.

---
