"""
Stochastic differential equation family — skeleton (stream E-SDE).

The home for the SDE family base class.  Per the resolved noise contract
(ROADMAP §11, diagonal-Itô), a stochastic system will be defined by a
``_drift(y, t, **params)`` (the deterministic part, exactly like
``_equations``) plus a ``_diffusion(y, t, **params)`` returning one noise
coefficient per state component, each multiplying an independent Wiener
increment.  Solvers: Euler–Maruyama (order 0.5) and Milstein (order 1.0).

This module is an intentional placeholder so the family has a stable import
path (:mod:`tsdynamics.families.stochastic`) for the E-SDE stream to build on.
It defines no public symbols yet; the registry's family detection gains the
``stochastic`` tag when the base class lands (stream C-FAM).
"""

__all__: list[str] = []
