# Independent scalar response-credit diagnostic

`experiment.py` contains the complete NumPy REINFORCE simulation;
`_support/paper.json` contains its fixed protocol. This diagnostic is independent of
the phase executor, so it needs no game, reward or neural-policy classes.

Run `python run.py response-credit --help` from the code root.
