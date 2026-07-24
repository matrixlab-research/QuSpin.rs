"""Lattice-map helpers for the QuSpin compatibility namespace.

The maps are only frontend data. Symmetry validation, period discovery, sector
construction, and fermionic phases are handled by the Rust core.

The public helper shape is adapted from QuSpin under its BSD-3-Clause license;
the notice is distributed as ``quspin/LICENSE.rst``.
"""

from __future__ import annotations

from itertools import product

import numpy as np


class site_info_square:
    def __init__(self, Lx: int, Ly: int):
        self.N = int(Lx) * int(Ly)
        self.sites = np.arange(self.N)
        self.X = self.sites % int(Lx)
        self.Y = self.sites // int(Lx)

    @property
    def coor_iter(self):
        return enumerate(zip(self.X, self.Y))


class square_lattice_trans:
    def __init__(self, Lx: int, Ly: int):
        self._Lx = int(Lx)
        self._Ly = int(Ly)
        if self._Lx <= 0 or self._Ly <= 0:
            raise ValueError("square-lattice dimensions must be positive")
        self.site_info = site_info_square(self._Lx, self._Ly)
        sites = self.site_info.sites
        x = self.site_info.X
        y = self.site_info.Y

        self.Z = -(sites + 1)
        self.Z_A = np.asarray(
            [
                -(site + 1) if (x_site + y_site) % 2 == 0 else site
                for site, (x_site, y_site) in enumerate(zip(x, y))
            ]
        )
        self.Z_B = np.asarray(
            [
                -(site + 1) if (x_site + y_site) % 2 == 1 else site
                for site, (x_site, y_site) in enumerate(zip(x, y))
            ]
        )
        self.T_x = (x + 1) % self._Lx + y * self._Lx
        self.T_y = x + ((y + 1) % self._Ly) * self._Lx
        self.P_x = (self._Lx - x - 1) + y * self._Lx
        self.P_y = x + (self._Ly - y - 1) * self._Lx
        self._P_d = y + self._Lx * x if self._Lx == self._Ly else None
        self._P_e = (
            (self._Ly - y - 1) + self._Lx * (self._Lx - x - 1)
            if self._Lx == self._Ly
            else None
        )

    @property
    def P_d(self):
        if self._P_d is None:
            raise ValueError("diagonal reflection requires Lx == Ly")
        return self._P_d

    @property
    def P_e(self):
        if self._P_e is None:
            raise ValueError("anti-diagonal reflection requires Lx == Ly")
        return self._P_e

    def allowed_blocks_spin_inversion_iter(self, Np, sps):
        maximum = (int(sps) - 1) * self._Lx * self._Ly
        include_inversion = Np is None or (
            maximum % 2 == 0 and int(Np) == maximum // 2
        )
        for blocks in self.allowed_blocks_iter():
            if include_inversion:
                for sector in range(2):
                    yield {**blocks, "zblock": (self.Z, sector)}
            else:
                yield blocks

    def allowed_blocks_iter_parity(self):
        for px, py in product(range(2), repeat=2):
            yield {
                "pxblock": (self.P_x, px),
                "pyblock": (self.P_y, py),
            }

    def allowed_blocks_iter(self):
        for kx, ky in product(
            range(-self._Lx // 2 + 1, self._Lx // 2 + 1),
            range(-self._Ly // 2 + 1, self._Ly // 2 + 1),
        ):
            if kx == 0:
                if ky == 0:
                    for px, py in product(range(2), repeat=2):
                        base = {
                            "kxblock": (self.T_x, kx),
                            "kyblock": (self.T_y, ky),
                            "pxblock": (self.P_x, px),
                            "pyblock": (self.P_y, py),
                        }
                        if px == py and self._Lx == self._Ly:
                            for diagonal in range(2):
                                yield {
                                    **base,
                                    "pdblock": (self.P_d, diagonal),
                                }
                        else:
                            yield base
                else:
                    for px in range(2):
                        yield {
                            "kxblock": (self.T_x, kx),
                            "kyblock": (self.T_y, ky),
                            "pxblock": (self.P_x, px),
                        }
            elif kx == self._Lx // 2 and self._Lx % 2 == 0:
                if ky == self._Ly // 2 and self._Ly % 2 == 0:
                    for px, py in product(range(2), repeat=2):
                        base = {
                            "kxblock": (self.T_x, kx),
                            "kyblock": (self.T_y, ky),
                            "pxblock": (self.P_x, px),
                            "pyblock": (self.P_y, py),
                        }
                        if px == py and self._Lx == self._Ly:
                            for diagonal in range(2):
                                yield {
                                    **base,
                                    "pdblock": (self.P_d, diagonal),
                                }
                        else:
                            yield base
                else:
                    for px in range(2):
                        yield {
                            "kxblock": (self.T_x, kx),
                            "kyblock": (self.T_y, ky),
                            "pxblock": (self.P_x, px),
                        }
            elif ky == 0 or (ky == self._Ly // 2 and self._Ly % 2 == 0):
                for py in range(2):
                    yield {
                        "kxblock": (self.T_x, kx),
                        "kyblock": (self.T_y, ky),
                        "pyblock": (self.P_y, py),
                    }
            elif kx == ky and self._Lx == self._Ly:
                for diagonal in range(2):
                    yield {
                        "kxblock": (self.T_x, kx),
                        "kyblock": (self.T_y, ky),
                        "pdblock": (self.P_d, diagonal),
                    }
            elif kx == -ky and self._Lx == self._Ly:
                for diagonal in range(2):
                    yield {
                        "kxblock": (self.T_x, kx),
                        "kyblock": (self.T_y, ky),
                        "pdblock": (self.P_e, diagonal),
                    }


__all__ = ["site_info_square", "square_lattice_trans"]
