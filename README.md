# leekspeak
*Luckily, I speak leek.*

### Idea

Onion addresses are hard to remember and look scary! Language is (sort of) easy to remember and looks friendly! I speak English, so I'll find a mapping between (likely nonsensical) English 'phrases' and onion addresses.

*(This could definitely be done for essentially arbitrary languages, provided constraints (see below) can be met.)*

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

### Implementation

    lol python

Getting the map is my primary concern. Given that, translating back and forth is extremely trivial. 

To be genuinely useful one probably needs a browser extension to do this automatically (although always with the option to view either version, for maximum safety!), but I don't know how to do that and am not sure that writing extensions for the Tor Browser is a good idea. 

![not a fix won't bug](https://dl.dropboxusercontent.com/u/1333033/dealwithit.gif)
