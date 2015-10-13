# leekspeak
*Luckily, I speak leek.*

---

### Idea

Onion addresses are hard to remember and look scary! Language is (sort of) easy to remember and looks friendly! I speak English, so I'll find a mapping between (likely nonsensical) English 'phrases' and onion addresses.

*(This could definitely be done for essentially arbitrary languages, provided constraints (see below) can be met.)*

---

### Background

Onion addresses (currently) consist of 16 base32 characters (it's the first 80 bits of the SHA1 hash of the private key, I think). Base32 is 2-7 and all letters, so 32 characters (surprisingly enough).

To encode this into English, I could use...

- 8 words (from a dict of 32^2 = 1024) (maybe too small)
- 4 words (from a dict of 32^4 = 1048576) (maybe too big... do people know this many words?)

A nice intermediate size would be
- 16/3 words (from a dict of 32^3 = 32768)
... but 16/3 is not a nice number.

If I expanded it to 18 characters (i.e. absord the 'on' at the end) we could get
- 18/3 = 6 words (from dict of 32^3 = 32768)
... which could be a workaround.

So, __I want a map from every
xyz (in base32) ---> English__

... but I don't want an *arbitrary* one.

---

### Constraints

#### Idea

    <user> here's a totally sweet onion service! http://2v7ibl5u4bpemwiz.onion/  
    <user> wait sorry I meant http://2v7ibl5u4pbemwiz.onion/

Can you see the difference? These addresses look pretty similar, and a malicious user could make a realistic-looking cat facts onion site and steal all your infos. (Assuming you were in the habit of giving infos to a cat-facts site...) You'd have to carefully check the address to defend against this, and that requires remembering it accurately (hard).

The attack relies on creating a *similar-looking* name, in base32 space (assuming a user will notice a *drastic* change...). But in base32->English space, it might look quite different. This would make detecting this attack easier for the user.

But what does *similar-looking* mean in English space? Two English words might be 'similar' if
  1. they share lots of characters (e.g. `bed` and `bad`, `yes` and `yea`)
  2. they share meaning (e.g. `Monday` and `Tuesday`, `cold` and `icy`)
  3. ???

Why is sharing meaning relevant? Suppose you don't *quite* remember the name of the onion site (in English words). It's probably about 6 words long, so that's fair. Maybe it has a day of the week in it, or something? Some sort of fruit... maybe an apricot? Or a peach? You make your best guess and go for it. The page loads and it *looks* like how you're expecting, so all must be well?

The problem is the (admittedly very slight) chance that your attacker's small base-32 modification induced a change from `apricot -> peach` in English space. Semantic similarity has wrought disaster upon you, and all of your infos are taken (or worse things). What can be done?

#### Solution

First of all, comparing the character content of words is easy, since it's right there. Phew.

Secondly, semantics! Distributed representations of language allow for relatively-accurate measures of *semantic relatedness* between words! All it takes is training a model on a large corpus of text.

Then we just need to select a map such that small distance in base32 space means large distance in English space, where we can measure this distance using some hand-crafted function combining *string features* (such as edit distance) and *semantic features* (such as distance in the vector space in which words obviously live).

#### Why English?

Pretty good word representations have already been obtained for English (although better ones presumably exist :)). I can repeat this for any language where I can obtain a large corpus. I'll just start with English since it's at hand.

---

### Implementation

Getting the map is my primary concern. Given that, translating back and forth is extremely trivial. 

To be genuinely useful one probably needs a browser extension to do this automatically (although always with the option to view either version, for maximum safety!), but I don't know how to do that and am not sure that writing extensions for the Tor Browser is a good idea. 

#### Current Status

Translation is fine, although probably requires some more/any error handling/better input parsing.

Before I do the fancy word vector stuff I'm using an arbitrary map between base32-triples and words, using a set of the most frequently-used English words (source: http://norvig.com/ngrams/, thanks 0xcaca0 for the link), to make sure everything works as expected.

Fancy word vector stuff has been partially developed using a heuristic approach I devised on an airplane, and is likely suboptimal (although I suspect the problem is NP hard). The chain of reasoning goes as follows:  
- requiring `small distance in base32-space == large distance in language space` basically means we need the pairwise distance matrices in each of these spaces to be as _different_ as possible
- that means, we probably want to maximise the mean of their absolute difference _(although whitening the data before doing this might make more sense)_
- the permitted operation to achieve this is moving any row+column to another row+column in the second matrix (second matrix being pairwise distances in the language space, although this choice is arbitrary) (I say row+column because the distance matrix is _pairwise_, but our mapping is on individual elements... so moving `j -> j'` will move _both_ *row* j and *column* j, both of them involving `j`)
- if we fix a column (or row, whatever) in both matrices and look at the _inner product_ between these, we would like to find a re-ordering of the elements in the second vetor (derived from the second matrix) to _minimise_ this inner product. I expect this to maximise the difference since these matrices are _positive_, and their difference will be greatest when they are as 'mutually exclusive' as possible, and the inner product will capture this mutual exclusivity. Of course, when we do the re-ordering we aren't allowed to move our 'fixed' column/row index, since that would mess everything up.
- do this iteratively, sort of like Gibbs sampling but probably much less theoretically sound
- fearing local minima, I threw in some stochasticity by proposing a reordering (using above procedure) and then accepting it if it increases the difference between our matrices, and accepting with some rejection probability related to how much it _decreases_ the difference... that is to say, Metropolis-Hastings has been dubiously invoked
- iterate until ???

This approach has yet to be tested rigorously. So far, it approximately works (in that the distance between the matrices mostly increases), but convergence is far from guaranteed, it seems somewhat sensitive to initial conditions, and the situation of global versus local extrema is unknown. It's also probably somewhat slow and memory intensive (woo, getting a 33,000 x 33,000 pairwise distance matrix between 100-dimensional vectors!), but that's surmountable. :)

---

### Problems

##### Won't attackers just go for the language representation, then?
Potentially, but this will force a large distance in base32-space, which the user will hopefully identify. Can't win them all (probably).

##### Appending an 'on' to get to a 18-character name is idiotic/something about entropy
Maybe, yeah. I just thought of this.

##### ???
I'm not a cryptographer.

##### Vector spaces? Distance metrics? This is overkill
This is related to my research, so it's pretty easy for me.

---

### Gif of a frog
![not a fix won't bug](https://dl.dropboxusercontent.com/u/1333033/dealwithit.gif)
