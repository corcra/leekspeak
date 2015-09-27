.onion addresses normally should be made of a base32 string of the first 80 bits of the SHA1 hash of the private key of the server
base32 is 2-7 and all letters (so 32 characters)
http://blacksunhq56imku.onion .... e.g.!
... so 16 base-32 characters!

To encode this into English, we could use...

8 words (from a dict of 32^2 = 1024) (maybe too small)
4 words (from a dict of 32^4 = 1048576) (maybe too big... do people know this many words?)

A nice intermediate size would be
16/3 words (from a dict of 32^3 = 32768)
but 16/3 is not a nice number.

If we expanded it to 18 characters (i.e. absord the 'on' at the end) we could get
18/3 = 6 words (from dict of 32^3 = 32768)
... which could be a workaround.

In this case we need to find a map from every
xyz (in base32) ---> English

But we want to enforce *some* constraints.
