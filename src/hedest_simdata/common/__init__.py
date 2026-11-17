"""Helpers shared by more than one family.

Nothing lands here speculatively: a module appears when the second family needs it, and
when the behaviour of the two call sites has been shown to be the same. :mod:`lattice` is
the cautionary example -- the three spot grids looked interchangeable and were not.
"""
