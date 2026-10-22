# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib.util

from tokenspeed_kernel.platform import current_platform


def _flash_mla_function(name):
    if (
        not current_platform().is_hopper_plus
        or importlib.util.find_spec("flash_mla") is None
    ):
        raise RuntimeError(
            "This attention implementation requires the optional flash_mla package"
        )
    import flash_mla

    return getattr(flash_mla, name)


def flash_mla_sparse_fwd(*args, **kwargs):
    return _flash_mla_function("flash_mla_sparse_fwd")(*args, **kwargs)


def flash_mla_with_kvcache(*args, **kwargs):
    return _flash_mla_function("flash_mla_with_kvcache")(*args, **kwargs)


def get_mla_metadata(*args, **kwargs):
    return _flash_mla_function("get_mla_metadata")(*args, **kwargs)


__all__ = ["flash_mla_sparse_fwd", "flash_mla_with_kvcache", "get_mla_metadata"]
