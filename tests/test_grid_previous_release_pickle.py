"""Pickles written by the previous release (pure Python grids) load into the C++-backed grids.

The oracle is the previous release itself (the Python-and-Numba implementation at commit
11ab436 of the main line): each pickle below was produced by running that release once, at
protocols 2 and 4, on the objects ``_make_objects`` rebuilds here, then compressed with zlib and
base64-encoded. A pickle of a class with ``__slots__`` stores the slot values, so these
loads exercise the ``__setstate__`` of each class: it must rebuild the compiled value from
the previous release's slots (and drop the lazily built spatial index it also pickled). The
loaded objects are compared with the same objects built by this release, in value: blocks,
version, tag names and arrays, breakpoints, node arrays.
"""

from __future__ import annotations

import base64
import pickle
import zlib
from typing import Any

import numpy as np
import pytest

from pantr.grid import (
    BVH,
    CellTags,
    FacetTags,
    HierarchicalGrid,
    Partition,
    TensorProductGrid,
    uniform_grid,
)

# name-pN -> zlib-compressed, base64-encoded pickle at protocol N.
_PREVIOUS_RELEASE_PICKLES: dict[str, str] = {
    "tpg-p2": (
        "eNqdVWl320QU9ZbEUQoNZS1rKQVcFkPZGqBg00JbKjBlCEWsQpLlSNSWfKVxaYDQlNZOwr6WHcLyA/KVwwc+8MNg"
        "3khyXLk5DsjnSNbTzJ17333zZjFntQ2PB+W5wK2XdW57oR/o7cCvdyyuU1CZlbGjUegQRZDZfRrZ2gJyJW0ik8no"
        "pt/x6iHyltdptefLuuUHdrnVaXLXCAJjXtED2/K9kAcCQUEhGqZ49egrxtRMF+OWmFa3rVCxPfqjYELLCnATRW1c"
        "PJsGd709mOxBYZhawhaGi0pqVs2puR4ujiHrfL4tZm7VcmJGYwbTK8tLuIRhW0nNS7R9uLRWqx35R1zypmY4LjNX"
        "nAltRyZ9FaupQBWXO2L9Kxiu5LjK1LZI7YFtHG/7rsdDbHcKjhRztSMYXsNwLTHMd3Gds43W2D60xt9rlWQxXE/g"
        "NzDs4LjRTJB2EtJNDLsIqdDFzRHSzmGk1UoKUvC9hSBvZShx7DZ7uE3bSpQtu9kM9bYd6MZJN8TtIof5Hu7Qpuij"
        "G+odz234QQt3rmhjFPLqbgtlNacp8q3TihBwlzquFSkknHXrdoi7E9Z7iPU9DPdKh7q4z4kscWdwP1nyAMNeYYkz"
        "bMaMNGNajM3HKrLxEw+SmIcYHubYZ2oFmfwTDh45r4JFRNl/7LCCR0WNVkSNVksxbVFTkjceS1juJ5YHGB4nllNd"
        "POHspcUXKYl//bdf34VUPCGfi5/5DcYV4vhYIvYgiT3EcJjjSTPKslTguDiS8FeJ/1MMT0v+tA9qUXX8MVzD1aHi"
        "GDHgQvWUHrJWGYHaB0kvU6wO1X+/aJ8h6UcZnuVg5oB5TbvB8VwifpbEP89wLDbvhci85YGKSZI9sUHSxzaIj/pN"
        "bnIeNFLyIsNLHC8Pmtj08Uqi41XS8RqD3jfx9cjEPzObvdbTP2rgamVUYNjSzMjCuEBkrQKDxJsMFkfdjHqLFB+4"
        "cw6HnehvkP45Bif20V33Mdk0yeYobpD38f/po7JZH98gKccZmhwtUx55XtwCPdECo3fSFsJXp2Rnki2zreY6PSCu"
        "YZqgc2MuRHD+mStCygHxcZb+IBR9i4u+1Sk51F9lA5azTizgTXmGGTiZJG+ekvcWw9txn30nSt70wCZIkocFEvEu"
        "wymOxf7xcpoA3mM4EwOcXQeYTO0edAmgx7DEsSwOk5VQiHs/drZhWDaPeH4wrO4gfY3kfSjkfSTkfVySi0QTo8NI"
        "NudP1ALpdoTaT6XaBj5LyH5OZL9g+LJ/7n8V0R08u7OpTpoUEc4R/a8ZvuH4tq//O4L8nuGHWP+P6/rzqS6Cnwjg"
        "Z4ZVjl+E/l9J/2+muP1ulv8F8Fy70w=="
    ),
    "hg-p2": (
        "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"
        "Dv4XnpQdIA=="
    ),
    "bvh-p2": (
        "eNqdlElv00AUxx1nrcNa9r3sKYuhrAEKLoUDwlIOc0C+RfHSjiXHyd+ZIOWAVA5N22/Rj5Ar4sAhHwxmnHFSUqy0"
        "zEie8Zs3z/N7/zfeUJ12I2SRvh75rl63v1Jt9csnDcrid2Rq36BWLE1RlHrYcr264wUBsk7YbbZ7et1pRZ7e7AbM"
        "b0RRo6fVI89phR0WdR2mITdy00J3tIq8qWyi4PBtrud0NC8UEw1FK8Pj2yhZBT4GDeaHS5jrQyMob+EIwdGKmTHL"
        "mzgmA7qs1+b7jlsq9/erOLGzvYWTBPMVMxvHWsapWq32+Tdv8cNUGE7bO7RobfDV4a/DdUW2aXtG2lU5ZlP8ctKe"
        "lyPOUE53luAcw3nbKo2TS31coDkap+ki5eyXCC7H7KbaxxU6Al6r4qoAvkawwIHpftTrMeqP5NxKaUVO5DgcGH8b"
        "Uh2Gu0bKlonLwJgRdRxk+jOl6VgTB9wQObpJcIvhtr2nAANvjeFOkqW7IksVgkVZIffovEDf5u6ZKVWKKerkU+yz"
        "+twB9+G+IHlA8JBB36t20MKjhOOx4FgieDJW+yldECQ/lYO2SfpnOe4aswz7JVVmFsY/LAMDzwT8c4IXDC9tqzyG"
        "j/x1ylBN+F8J/tcEb6SOyxMdk9uV3KJSSt4L/6mjdlAd3wqUdwQGw4ptiXIK499hB+/NgnwXbB2smmVLHDd0/SY+"
        "mGq3j4+2/gfSiq3G"
    ),
    "part-p2": (
        "eNo9UMtOwzAQdNLSFpc35fkF4RIJiSNfgKUI5eRblDgWtZQ6GccRygEJDm3pV3Hgw8BJELPS7kg7u6Pdd19UqbYm"
        "fDEqD5MqNVZZVWr6/M9A7j7gRW/wAz4nhCRCFkVSvmppMBK6WVVtmIjSyHDVFFalxqQtTYwUpa6taYSlGA8yqvOh"
        "iz1G1pgIN5ZLUVOpO0Ix5Z4zyDDjE1eL1Cp9j/0NaIz5FgcxDgPmsfEaR38Lc9tWbu6Y+06vHnCy+9ziNMZZwEb9"
        "rkecR1H09OPQJ0YsFtluOeULMqBTfX8N0XFcLJ3jZYwri+uMz7qLdf+YGjfMbza4zcJfwW9kVQ=="
    ),
    "cell_tags-p2": (
        "eNpVjktPg0AUhQewVMdXfVTrsz4T3JC4c+HOnSQsiAt2BAZSSCjtGYYFJk100db+Khf+MB0oxnhvMvfO3HO+O28q"
        "G/uZ4OaAJ6HpCX+Q06coTV+qBuTuHYo9gWq4lBDiZcXQY3KaQ7N0t1U9VQ6sTNByFXl9hc6kaFyaHhvxyBwWqUh8"
        "zv2SejxioywXvGCCor2U0SxcTrFqkSnWmLSFEctplFUNBa2xAdZdXdbUF0l2j40ZNh1szbHtoGNYiqVOsdMAQ1GO"
        "pW/XVaU+ecDe4mOOfQddw9Jq1iMObNt+/pZRHxYROAwWMXU75C+0pqIXy21HDo4FToK4Hdf/PI3l8jMH583yftyt"
        "AD1p+Pr8n+ov6KICXTq4ErgOZrjJixluA/MHQ+R2sA=="
    ),
    "facet_tags-p2": (
        "eNplkLtPwzAQxtP0QevyKK/ypuWpskRiY2BmIFKGiMFblDihiZSm+RxHqEMlGJrSvxrsNEzc4Dvf3ffzJ3/qLHUT"
        "wY0xj3zDEe44Iy8uC8SbqqA9fKFmzaGPaE/TNOddjTInDbjDgjhG3WxQogZJPik7GRpmnTZVS7HQnKNFa/L6gQ0m"
        "l9KZ4bApD4xJHovI5dydEYcHbJpkgudMELTXayTx11N0TG0BwqTMD1hGgkQVBN0S62GTtmSOXRElj9gqsG1jZ4me"
        "jd2RWTN1Uy+wVyF9MUulcp/qUhE94WD1vcShjf5Iela0ZxxZlvX6I6M8TE3g2FuFXTrU/ode5XqVcRLK909tnAmc"
        "e2E7LJ1fhNLOpY1BaWeBYdhXQPWdnUrY/gNcKcC1jRuBW6/AXZYXuPeMX1XqczQ="
    ),
    "tpg-p4": (
        "eNqNVN2K00AUzu823ShWwcU7vVxBerUXXggtFlQMFJFFvBvSJmnCtkmZJIu9WNALq4W52+N7+Eh9hH2DdX6SNZmm"
        "6R5oM+dv5pvvnDnfjT/djsKFPF+6cYb7Mxx5fZT5cZpgtMSJl08zxIxAHp9z6ydhfM9s1/DyB4yv4JR00CTJYy8F"
        "chLni+Wqj6YJ9vuLfJ5FLsbuCsgDhP1pEqcZpuk0lZg8Ekgn9kTINTjKTxipE/gFn+HUUR3N0dYQdonpZaulz5K0"
        "4DVsfgu/TtQ3MB6PP95S4X+OksFkM3qh1MQa1nVlCDSMAppg371YJlGcpRBaoc2ODx+Wh+tUeboZPavn3gzKTdkm"
        "u1mGyJIgbAf1bAFhDeQRmvrzeYqWPkbutygFemed2m0UpSiPoyDBC9gQE8VetKBOcowobSIJnCNiIUpo5PlNF9Co"
        "8oRoUYWx8KSJr55eoFKLL+fHQJPLkCKsdga36G+/fKjUniJKPJ9DagBhU+VsM/p6u0dKhkq9hKAVX13yG4VuVqFa"
        "AkIYNQHgPURL8lfqhqFcknb/bgnr+s2gfb/toHl/ayg3Vq1LC3bnfpC1sKtKbHUk1kxJl6W7x19ld57cg91DcsfO"
        "AdkO2vUdtpX2am13qsSvZour4WgWttFbNmPZfJZE19EBeo/b6O3E/58zXTNAdG0To3jy+Zq3AYtBmTujM7ZXG9bc"
        "ZI2o+5wt7x5m+IpuaAr/FRDV3TMi6AV7qnTB5tlWRnelLhOzLOVAbRS4Uz8Tx4YB6b5jah0Z6YkgMfb44HAMBje8"
        "YECDJqC8z84qg1WVBoF2L+i69FAq0Nmv/w/OsSdk"
    ),
    "hg-p4": (
        "eNqdWN1v2zYQl2w5dhK3UZp+ZB9ds3Zb065zMGAD+jAgRgxswQQEw1DsJRgI2aYjIYpkSHKQDCiwPSSrAb6V+1v2"
        "sj/KD3nIQ4COpKhMPotmOiIxdfc7Ho+nuzPPv1t/PV4zxCAPh26Yxq2D2O+3kOfj2I17nt9zA8RZlNi7Bd4PnPWW"
        "PvuD7r2mm6SOukHUO0zor3ST/TmGY5xTx3Sq51TMnKo6pqQqgsqwqsCsjMJ8bYU/M44lcEtIL2QUwzPAaTgLnOFi"
        "tvXA7aVRTNlCJkqWUYCPcYC6boJpZsuSYzvrmCyhcHSEejgIEuqskzto6PYOcT8zPZMn95nI8LSFelGMW0ejIPXd"
        "OHZPKWmiGPeiMEnjUS9lJyc1IUlJPexnIm+pY5zRjtmlf9Kf2b6mUz+j3iap9dPTIeZLKv5LOn6ToVVifkf39vZ+"
        "fMeG+HCMlHbHnZeGHFU51+Rcl/OSnJtytuVM2XJiTx/K86n31HvODfNeXJvF/eR9M+4MTbBVPqv4loLO5wVANwC/"
        "zMQgmmuiAYapoCvANEPBtwAfHkGYuJabmIVSkrpxWmKldSZs3DAUbqgXddZQHEUpJY+KaZbiMIliNIyjPgsrmWir"
        "rwT3p4w5m2nRKOwnJeaI+Pe+JpVBIcy8b8uCbGPaSXZ7mm60hclN1I2xeziM/DAt27DGiP1xZ3N68WRbPrRzZfkm"
        "XGm5F/dnTJrVwlez/F7JMhgNcYzcEz+RxYPlvZ+gUegPoviIvmHuDvv+ESsK3hOnSRrsHTI/4lK3ZW/RhsEvXGCh"
        "7rFH91jt4Lui1D1IGLXMaw5OJTk6Z4Kr4vWiw5B9iHiZt5UBUlvmxbWGBA0CtyziFjNf7d7U4zfxKXPOMY4TPwpZ"
        "NfZOyEoxQPnpSXXnl91CCLI6GvWxcEiJiVvijDsnzI3v/ufI7c3p3F+rIJdz/I6k7wJ+PkzAr4BKugbwe+D16Oy9"
        "D+QegAptAfw2qA9Q3zLg3wJ1pQHwFUkvFsOpkb2l0m+ALVEp9sc7kwosAXnsqGIJ4peK2LvYLsdz/gTMUP5CMUN9"
        "+forjR4VrjovzBUoPwH0pUJe5Z8rDW4DPdD/lgJX+dvS2GlpzmkrcLg/9IsufuC54flUuA1wWyE3i4vskDUswIN0"
        "fg2DV6EPQG1YAbkMc3dRU0NuKfC81t0D+JpG3wMF/pHkfy7nh3L+FMg/0uh/osCfSv6XAH+m0feVAi/WsNIrYkkN"
        "043r2NeMyfb78SGum2ENzMeFRu5SsU6lX2X3jB/a5ftcbM+361J1HlDj4L6wZsN1+bjSyMOabrTn67MUNKxZ0F5Y"
        "I6GfVDVVRau+a6wb1mQjq2HLWXbE/oGnKWKw2fkQXChscLFogqRc0iTxbQWeX8zgBemuRt+6Av9Y8r+Q8ydyfgzk"
        "NzT6P1PgeSfzAuDPNfpa84pYPfzvNwf2zN8Ze94ilmxQePvgnRJ7qi/kjQVpdNi6V/zx+vLNupl11tsInN/FTfc9"
        "2wx9/1PshFlfZXZLdjCzNY3KXM25lDmlcSQO/Jv3N1n8nndR0wdkbZDorbIGT/QYjsVP7f3zmhkzUPa9xUa8qvix"
        "YL4DFkrcdU4Tbi7/b/0LfBM2fQ=="
    ),
    "bvh-p4": (
        "eNqFk81Og0AQx4GWQovGRGPv3uqFswcTjL0YSTh4MN5I+WiXhEKzBZMeTPTQapO9dXwPH6mP0DeofCy1boH+D8xO"
        "Zmcy85vlvfl90eQykbPJIIiwOsKeo5rWKwLSuH9+gBVcf4DxBj3SMYPQcU3b9X0g3SAeT2aqaYfYVcexH3kDjAcz"
        "ICcmdu0wmEY4tqMkm4jZTSBS4ORXVqBzc+jzFnzCE/R0XlfmgNpEdKLZxE1TBO8Gll95tEH4WzAM43GbKPvoXATW"
        "sv+yrRBHVfg89QVqG0ycAuBEaiEpT+R8WOQBkpGSNoxOd+3qwgLQORGGe22iblmTP3lJ+Y7Wpnaj/fer4mut/P5G"
        "K7dsvSKfrS8zdf7i2fB00747jMrGT5zLhD/P8JQYriLjs2pXxPf5+2El/6sd3WPa0TmitVbvH9Dm6re1PthSNpqS"
        "j4a9EarDWzzX4nnKDK7WEbydOrxSkP3HU9Bb6TltKDkrpBk43hh0IV6Apf4CiP5NLQ=="
    ),
    "part-p4": (
        "eNo9T7tqw0AQlGQL2yRFmvgb5OaqlOnc+UCY/MChx5EcyCexPhFUGJzCTgzbafKJ/g5bD/DA7iyzM8Ucp/9XbwS/"
        "Vol1JD7J5EJVCTnjTGnBi+3jbrH6QXxAxE8q00Whym+rCby09a5qhMpK0mJXF84kREkDflaks9LuHdWZ6+IcDk7w"
        "zOajpYX0Tlj7KX7xgUj6cnrC14LD3DWV7iOBecPlb/xO2H9HHMebW4dhSc8hvaxf+gp+N73WMzqZ58oOTfaQQX1G"
        "Ku7YW18u"
    ),
    "cell_tags-p4": (
        "eNpNjDtvwjAUheNAWkQRijqg/gRYPDN0y1ZLGSp2Ky+RSCFEjjMECakdoEW6G4ffC85j4BvutXy+e37Gt3dmdZBb"
        "BoVWfKuymEsdbCvQxEvyfNM+r1j9wj9iSVNZ1DsZmaCCeCGnV48gdgAtTFY2XEZ7lfBdnessUCpoQDOpkmhfVFrV"
        "kTZt5HQm6LWIe+UKYZ3gsRB/+MZSMGGfkM7IiXVTJu2Jna1x+e/TEbFP+L7/dTd0Q1ga4cVzrYHRsGG+07d03ran"
        "7nP3h7HvA/aTfUZVnxHyB47UbAc="
    ),
    "facet_tags-p4": (
        "eNpdjTtLxEAUhTPJBl1ZMTaCneXapBaxC1g4kELshzzGTSCbDZMJkmJBC6OB2+3x9+rkgYinuM/v3vO6+Lpl1ijy"
        "qqjUyt+oPPWFjjY1aHkfJVI/DfUB128I91iTJ56HaS0qqUQiiwJ8QSeibLZjV4M75E4P9iD2Arowu6r1RbJT0t82"
        "hc4jpaIWtBJKJruy1qpJtLEgdyRBR2U6IQdw6x0Bi/GBR6w54za3O2Sn5Ka6reRwZOc36D+nvUPsDmEYPnwbjYFb"
        "GnEfXFn/ZM/ZmTMMlq2ys8EvO/91M81lH3jLmTr+Q3eomw6x/wMA0W0C"
    ),
}


def _make_objects() -> dict[str, Any]:
    """Build, with this release, the objects the previous release pickled."""
    t = TensorProductGrid([np.array([0.0, 1.0, 3.0]), np.array([0.0, 0.5, 1.0, 2.0])])
    t.cell_tags.set("a", [4, 1], [7, 9])
    t.facet_tags.set("f", [[0, 1], [5, 2]], 3)
    g = HierarchicalGrid(uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3]), 2)
    g.refine(0, [1, 1], [3, 2])
    g.refine(1, [2, 2], [4, 3])
    g.cell_tags.set("a", [0, 5], [3, 4])
    g.cell_tags.set("b", [2], 1)
    g.facet_tags.set("f", [[1, 0], [0, 3]], [5, 6])
    bvh = BVH.from_cell_bounds(*t.collect_cell_bounds())
    part = Partition([0, 1, -1, 1], 2)
    ct = CellTags(6)
    ct.set("z", [3, 0], [2, -1])
    ft = FacetTags(3, 4)
    ft.set("w", [[2, 3], [0, 0]], [8, 9])
    return {"tpg": t, "hg": g, "bvh": bvh, "part": part, "cell_tags": ct, "facet_tags": ft}


def _load(name: str, protocol: int) -> Any:
    """Unpickle the previous release's pickle of object ``name`` at ``protocol``."""
    raw = zlib.decompress(base64.b64decode(_PREVIOUS_RELEASE_PICKLES[f"{name}-p{protocol}"]))
    return pickle.loads(raw)


def _assert_tags_equal(got: CellTags | FacetTags, want: CellTags | FacetTags) -> None:
    assert list(got) == list(want)
    assert got.num_cells == want.num_cells
    for name in want:
        for a, b in zip(got[name], want[name], strict=True):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
            assert not a.flags.writeable


def _assert_grid_tags_equal(got: Any, want: Any) -> None:
    _assert_tags_equal(got.cell_tags, want.cell_tags)
    _assert_tags_equal(got.facet_tags, want.facet_tags)


@pytest.mark.parametrize("protocol", [2, 4])
class TestPreviousReleasePickles:
    """Each previous-release pickle loads and equals the object this release builds."""

    def test_tensor_product_grid(self, protocol: int) -> None:
        got = _load("tpg", protocol)
        want = _make_objects()["tpg"]
        assert isinstance(got, TensorProductGrid)
        assert got.cells_per_axis == want.cells_per_axis
        assert got.num_cells == want.num_cells
        for a, b in zip(got.breakpoints, want.breakpoints, strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(got.bounds, want.bounds)
        _assert_grid_tags_equal(got, want)
        # The grid is usable: the previous release's pickled spatial index was dropped and the
        # lazy one answers.
        np.testing.assert_array_equal(
            got.locate_many([[0.5, 0.25], [2.5, 1.5]]), want.locate_many([[0.5, 0.25], [2.5, 1.5]])
        )

    def test_hierarchical_grid(self, protocol: int) -> None:
        got = _load("hg", protocol)
        want = _make_objects()["hg"]
        assert isinstance(got, HierarchicalGrid)
        # Version 3 and these blocks are what the previous release held (two refines).
        assert got.version == want.version == 3
        assert got._blocks == want._blocks
        assert got.factor == want.factor
        assert got.num_cells == want.num_cells
        assert got.max_level == want.max_level
        assert got.root.cells_per_axis == want.root.cells_per_axis
        _assert_grid_tags_equal(got, want)
        pts = np.array([[0.2, 0.2], [1.3, 1.1], [3.9, 2.9]])
        np.testing.assert_array_equal(got.locate_many(pts), want.locate_many(pts))
        # A pickled grid is still a live one: refining it works and bumps the version.
        got.refine(0, [0, 0], [1, 1])
        assert got.version == 4

    def test_bvh(self, protocol: int) -> None:
        got = _load("bvh", protocol)
        want = _make_objects()["bvh"]
        assert isinstance(got, BVH)
        assert (got.ndim, got.n_cells, got.n_nodes) == (want.ndim, want.n_cells, want.n_nodes)
        for name in ("node_lo", "node_hi", "node_left", "node_right", "node_cell"):
            np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
        assert not got.node_lo.flags.writeable

    def test_partition(self, protocol: int) -> None:
        got = _load("part", protocol)
        assert isinstance(got, Partition)
        np.testing.assert_array_equal(got.cell_owner, [0, 1, -1, 1])
        assert got.cell_owner.dtype == np.int32
        assert got.n_parts == 2
        np.testing.assert_array_equal(got.owned_cells(1), [1, 3])
        np.testing.assert_array_equal(got.active_mask, [True, True, False, True])

    def test_cell_tags(self, protocol: int) -> None:
        got = _load("cell_tags", protocol)
        assert isinstance(got, CellTags)
        _assert_tags_equal(got, _make_objects()["cell_tags"])
        np.testing.assert_array_equal(got["z"][0], [0, 3])
        np.testing.assert_array_equal(got["z"][1], [-1, 2])

    def test_facet_tags(self, protocol: int) -> None:
        got = _load("facet_tags", protocol)
        assert isinstance(got, FacetTags)
        _assert_tags_equal(got, _make_objects()["facet_tags"])
        assert got.facets_per_cell == 4
        np.testing.assert_array_equal(got["w"][0], [[0, 0], [2, 3]])
        np.testing.assert_array_equal(got["w"][1], [9, 8])
