# pylint: disable=C0302
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=unused-argument
"""Namespace for numpy operators used in Gluon dispatched by F=ndarray."""

from __future__ import absolute_import
import numpy as _np
from ...base import numeric_types
from ...util import set_module
from ...context import current_context
from . import _internal as _npi
from ..ndarray import NDArray

__all__ = ['zeros', 'ones', 'full', 'add', 'subtract', 'multiply', 'divide', 'mod', 'remainder', 'power',
           'arctan2', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'log10', 'sqrt', 'cbrt', 'abs',
           'absolute', 'exp', 'expm1', 'arcsin', 'arccos', 'arctan', 'sign', 'log', 'degrees', 'log2',
           'log1p', 'rint', 'radians', 'reciprocal', 'square', 'negative', 'fix', 'ceil', 'floor',
           'trunc', 'logical_not', 'arcsinh', 'arccosh', 'arctanh', 'tensordot', 'histogram',
           'linspace', 'expand_dims', 'tile', 'arange', 'split', 'concatenate', 'stack', 'vstack', 'mean',
           'maximum', 'minimum', 'swapaxes', 'clip', 'argmax', 'std', 'var', 'indices', 'copysign',
           'ravel', 'hanning', 'hamming', 'blackman', 'flip', 'around', 'hypot', 'rad2deg', 'deg2rad',
           'unique', 'lcm', 'tril', 'identity', 'take', 'ldexp', 'vdot', 'inner', 'outer',
           'equal', 'not_equal', 'greater', 'less', 'greater_equal', 'less_equal']


@set_module('mxnet.ndarray.numpy')
def zeros(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `zeros` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.zeros(shape=shape, ctx=ctx, dtype=dtype)


@set_module('mxnet.ndarray.numpy')
def ones(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with ones.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `ones` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.ones(shape=shape, ctx=ctx, dtype=dtype)


@set_module('mxnet.ndarray.numpy')
def full(shape, fill_value, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return a new array of given shape and type, filled with `fill_value`.
    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array. The default, `None`, means
        `np.array(fill_value).dtype`.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    ctx: to specify the device, e.g. the i-th GPU.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.
    Notes
    -----
    This function differs from the original `numpy.full
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html`_ in
    the following way(s):
    - Have an additional `ctx` argument to specify the device
    - Have an additional `out` argument
    - Currently does not support `order` selection
    See Also
    --------
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.
    Examples
    --------
    >>> np.full((2, 2), 10)
    array([[10., 10.],
           [10., 10.]])
    >>> np.full((2, 2), 2, dtype=np.int32, ctx=mx.cpu(0))
    array([[2, 2],
           [2, 2]], dtype=int32)
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.full(shape=shape, value=fill_value, ctx=ctx, dtype=dtype, out=out)


@set_module('mxnet.ndarray.numpy')
def arange(start, stop=None, step=1, dtype=None, ctx=None):
    """Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval. The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values. For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array. The default is `float32`.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    if start is None and stop is None:
        raise ValueError('start and stop cannot be both None')
    if step == 0:
        raise ZeroDivisionError('step cannot be 0')
    return _npi.arange(start=start, stop=stop, step=step, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def identity(n, dtype=None, ctx=None):
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``numpy.float32``.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> np.identity(3)
    >>> np.identity(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    if not isinstance(n, int):
        raise TypeError("Input 'n' should be an integer")
    if n < 0:
        raise ValueError("Input 'n' cannot be negative")
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.identity(shape=(n, n), ctx=ctx, dtype=dtype)


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def take(a, indices, axis=None, mode='raise', out=None):
    r"""
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : ndarray
        The source array.
    indices : ndarray
        The indices of the values to extract. Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype.
    mode : {'clip', 'wrap'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'clip' -- clip to the range (default)
        * 'wrap' -- wrap around

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray
        The returned array has the same type as `a`.

    Notes
    -----

    This function differs from the original `numpy.take
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html>`_ in
    the following way(s):

    - Only ndarray or scalar ndarray is accepted as valid input.

    Examples
    --------
    >>> a = np.array([4, 3, 5, 7, 6, 8])
    >>> indices = np.array([0, 1, 4])
    >>> np.take(a, indices)
    array([4., 3., 6.])

    In this example for `a` is an ndarray, "fancy" indexing can be used.

    >>> a[indices]
    array([4., 3., 6.])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(a, np.array([[0, 1], [2, 3]]))
    array([[4., 3.],
           [5., 7.]])
    """
    if mode not in ('wrap', 'clip', 'raise'):
        raise NotImplementedError(
            "function take does not support mode '{}'".format(mode))
    if axis:
        return _npi.take(a, indices, axis, mode, out)
    else:
        return _npi.take(_npi.reshape(a, -1), indices, 0, mode, out)
# pylint: enable=redefined-outer-name


#pylint: disable= too-many-arguments, no-member, protected-access
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None, out=None):
    """ Helper function for element-wise operation.
    The function will perform numpy-like broadcasting if needed and call different functions.

    Parameters
    --------
    lhs : ndarray or numeric value
        Left-hand side operand.

    rhs : ndarray or numeric value
        Right-hand operand,

    fn_array : function
        Function to be called if both lhs and rhs are of ``ndarray`` type.

    fn_scalar : function
        Function to be called if both lhs and rhs are numeric values.

    lfn_scalar : function
        Function to be called if lhs is ``ndarray`` while rhs is numeric value

    rfn_scalar : function
        Function to be called if lhs is numeric value while rhs is ``ndarray``;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    --------
    mxnet.numpy.ndarray or scalar
        result array or scalar
    """
    from ...numpy import ndarray
    if isinstance(lhs, numeric_types):
        if isinstance(rhs, numeric_types):
            return fn_scalar(lhs, rhs, out=out)
        else:
            if rfn_scalar is None:
                # commutative function
                return lfn_scalar(rhs, float(lhs), out=out)
            else:
                return rfn_scalar(rhs, float(lhs), out=out)
    elif isinstance(rhs, numeric_types):
        return lfn_scalar(lhs, float(rhs), out=out)
    elif isinstance(rhs, ndarray):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(rhs))))
#pylint: enable= too-many-arguments, no-member, protected-access


@set_module('mxnet.ndarray.numpy')
def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : ndarray
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details. The default is None.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.

    This function differs from the original `numpy.unique
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html>`_ in
    the following aspects:

    - Only support ndarray as input.
    - Object arrays or structured arrays are not supported.

    Examples
    --------
    >>> np.unique(np.array([1, 1, 2, 2, 3, 3]))
    array([1., 2., 3.])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1., 2., 3.])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1., 0., 0.],
           [2., 3., 4.]])

    Return the indices of the original array that give the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array([1., 2., 3., 4., 6.])
    >>> indices
    array([0, 1, 5, 3, 2], dtype=int64)
    >>> a[indices]
    array([1., 2., 3., 4., 6.])

    Reconstruct the input array from the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1., 2., 3., 4., 6.])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1], dtype=int64)
    >>> u[indices]
    array([1., 2., 6., 4., 2., 3., 2.])
    """
    return _npi.unique(ar, return_index, return_inverse, return_counts, axis)


@set_module('mxnet.ndarray.numpy')
def add(x1, x2, out=None):
    """Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be added. If x1.shape != x2.shape, they must be broadcastable to
        a common shape (which may be the shape of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    add : ndarray or scalar
        The sum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.add, _np.add, _npi.add_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def subtract(x1, x2, out=None):
    """Subtract arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be subtracted from each other. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape
        of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    subtract : ndarray or scalar
        The difference of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.subtract, _np.subtract, _npi.subtract_scalar,
                         _npi.rsubtract_scalar, out)


@set_module('mxnet.ndarray.numpy')
def multiply(x1, x2, out=None):
    """Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be multiplied. If x1.shape != x2.shape, they must be broadcastable to
        a common shape (which may be the shape of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        The multiplication of x1 and x2, element-wise. This is a scalar if both x1 and x2
        are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.multiply, _np.multiply, _npi.multiply_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def divide(x1, x2, out=None):
    """Returns a true division of the inputs, element-wise.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.ndarray.numpy')
def mod(x1, x2, out=None):
    """Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.ndarray.numpy')
def remainder(x1, x2, out=None):
    """Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.ndarray.numpy')
def power(x1, x2, out=None):
    """First array elements raised to powers from second array, element-wise.

    Parameters
    ----------
    x1 : ndarray or scalar
        The bases.

    x2 : ndarray or scalar
        The exponent.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        The bases in x1 raised to the exponents in x2.
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.power, _np.power, _npi.power_scalar, _npi.rpower_scalar, out)


@set_module('mxnet.ndarray.numpy')
def tensordot(a, b, axes=2):
    r"""
    tensordot(a, b, axes=2)
    Compute tensor dot product along specified axes for arrays >= 1-D.
    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an ndarray object containing two ndarray
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.
    Parameters
    ----------
    a, b : ndarray, len(shape) >= 1
        Tensors to "dot".
    axes : int or (2,) ndarray
        * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        * (2,) ndarray
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements ndarray must be of the same length.
    See Also
    --------
    dot, einsum
    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`
    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    Examples
    --------
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    """
    if _np.isscalar(axes):
        return _npi.tensordot_int_axes(a, b, axes)

    if len(axes) != 2:
        raise ValueError('Axes must consist of two arrays.')
    a_axes_summed, b_axes_summed = axes
    if _np.isscalar(a_axes_summed):
        a_axes_summed = (a_axes_summed,)
    if _np.isscalar(b_axes_summed):
        b_axes_summed = (b_axes_summed,)

    if len(a_axes_summed) != len(b_axes_summed):
        raise ValueError('Axes length mismatch')

    return _npi.tensordot(a, b, a_axes_summed, b_axes_summed)


@set_module('mxnet.ndarray.numpy')
def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):  # pylint: disable=too-many-arguments
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : ndarray
        Input data. The histogram is computed over the flattened array.
    bins : int or NDArray
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
        .. versionadded:: 1.11.0
        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.
    range : (float, float)
        The lower and upper range of the bins. Required when `bins` is an integer.
        Values outside the range are ignored. The first element of the range must
        be less than or equal to the second.
    normed : bool, optional
        Not supported yet, coming soon.
    weights : array_like, optional
        Not supported yet, coming soon.
    density : bool, optional
        Not supported yet, coming soon.
    """
    if normed is True:
        raise NotImplementedError("normed is not supported yet...")
    if weights is not None:
        raise NotImplementedError("weights is not supported yet...")
    if density is True:
        raise NotImplementedError("density is not supported yet...")
    if isinstance(bins, numeric_types):
        if range is None:
            raise NotImplementedError("automatic range is not supported yet...")
        return _npi.histogram(a, bin_cnt=bins, range=range)
    if isinstance(bins, (list, tuple)):
        raise NotImplementedError("array_like bins is not supported yet...")
    if isinstance(bins, str):
        raise NotImplementedError("string bins is not supported yet...")
    if isinstance(bins, NDArray):
        return _npi.histogram(a, bins=bins)
    raise ValueError("np.histogram fails with", locals())


@set_module('mxnet.ndarray.numpy')
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, ctx=None):  # pylint: disable=too-many-arguments
    r"""
    Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : real number
        The starting value of the sequence.
    stop : real number
        The end value of the sequence, unless endpoint is set to False. In
        that case, the sequence consists of all but the last of num + 1
        evenly spaced samples, so that stop is excluded. Note that the step
        size changes when endpoint is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, stop is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are num equally spaced samples in the closed interval
        `[start, stop]` or the half-open interval `[start, stop)`
        (depending on whether endpoint is True or False).
    step : float, optional
        Only returned if retstep is True
        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. , 2.2, 2.4, 2.6, 2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1.asnumpy(), y.asnumpy(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2.asnumpy(), (y + 0.5).asnumpy(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    Notes
    -----

    This function differs from the original `numpy.linspace
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html>`_ in
    the following aspects:

    - `start` and `stop` do not support list, numpy ndarray and mxnet ndarray
    - axis could only be 0
    - There could be an additional `ctx` argument to specify the device, e.g. the i-th
      GPU.
    """
    if isinstance(start, (list, _np.ndarray, NDArray)) or \
       isinstance(stop, (list, _np.ndarray, NDArray)):
        raise NotImplementedError('start and stop only support int')
    if axis != 0:
        raise NotImplementedError("the function only support axis 0")
    if ctx is None:
        ctx = current_context()
    if retstep:
        step = (stop - start) / (num - 1)
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype), step
    else:
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype)


@set_module('mxnet.ndarray.numpy')
def expand_dims(a, axis):
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ndarray
        Output array. The number of dimensions is one greater than that of
        the input array.
    """
    return _npi.expand_dims(a, axis)


@set_module('mxnet.ndarray.numpy')
def lcm(x1, x2, out=None):
    """
    Returns the lowest common multiple of ``|x1|`` and ``|x2|``

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays for computing lowest common multiple. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape of
        one or the other).

    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    y : ndarray or scalar
        The lowest common multiple of the absolute value of the inputs
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    gcd : The greatest common divisor

    Examples
    --------
    >>> np.lcm(12, 20)
    60
    >>> np.lcm(np.arange(6, dtype=int), 20)
    array([ 0, 20, 20, 60, 20, 20], dtype=int64)
    """
    return _ufunc_helper(x1, x2, _npi.lcm, _np.lcm, _npi.lcm_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def tril(m, k=0):
    r"""
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : ndarray, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle

    Examples
    --------
    >>> a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    >>> np.tril(a, -1)
    array([[ 0.,  0.,  0.],
           [ 4.,  0.,  0.],
           [ 7.,  8.,  0.],
           [10., 11., 12.]])
    """
    return _npi.tril(m, k)


def _unary_func_helper(x, fn_array, fn_scalar, out=None, **kwargs):
    """Helper function for unary operators.

    Parameters
    ----------
    x : ndarray or scalar
        Input of the unary operator.
    fn_array : function
        Function to be called if x is of ``ndarray`` type.
    fn_scalar : function
        Function to be called if x is a Python scalar.
    out : ndarray
        The buffer ndarray for storing the result of the unary function.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        Result array or scalar.
    """
    if isinstance(x, numeric_types):
        return fn_scalar(x, **kwargs)
    elif isinstance(x, NDArray):
        return fn_array(x, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
def sin(x, out=None, **kwargs):
    r"""Trigonometric sine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The sine of each element of x. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.sin, _np.sin, out=out, **kwargs)

@set_module('mxnet.ndarray.numpy')
def cos(x, out=None, **kwargs):
    r"""Cosine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The corresponding cosine values. This is a scalar if x is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.cos, _np.cos, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def sinh(x, out=None, **kwargs):
    """Hyperbolic sine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or ``-1j * np.sin(1j*x)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic sine values. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.sinh, _np.sinh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def cosh(x, out=None, **kwargs):
    """Hyperbolic cosine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic cosine values. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.cosh, _np.cosh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def tanh(x, out=None, **kwargs):
    """
    Compute hyperbolic tangent element-wise.
    Equivalent to ``np.sinh(x)/np.cosh(x)``.
    Parameters
    ----------
    x : ndarray or scalar.
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
       The corresponding hyperbolic tangent values.
    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)
    - input x does not support complex computation (like imaginary number)
    >>> np.tanh(np.pi*1j)
    TypeError: type <type 'complex'> not supported
    Examples
    --------
    >>> np.tanh(np.array[0, np.pi]))
    array([0.       , 0.9962721])
    >>> np.tanh(np.pi)
    0.99627207622075
    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = np.array(1)
    >>> out2 = np.tanh(np.array(0.1), out1)
    >>> out2 is out1
    True
    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> np.tanh(np.zeros((3,3)),np.zeros((2,2)))
    mxnet.base.MXNetError:
    [07:17:36] ../src/ndarray/./../operator/tensor/../elemwise_op_common.h:135:
    Check failed: assign(&dattr, vec.at(i)): Incompatible attr in node
    at 0-th output: expected [3,3], got [2,2]
    """
    return _unary_func_helper(x, _npi.tanh, _np.tanh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def log10(x, out=None, **kwargs):
    """Return the base 10 logarithm of the input array, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which t'absolute', he result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.
    Returns
    -------
    y : ndarray or scalar
        The logarithm to the base 10 of `x`, element-wise. NaNs are
        returned where x is negative. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.log10, _np.log10, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def sqrt(x, out=None, **kwargs):
    """
    Return the non-negative square-root of an array, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        The values whose square-roots are required.
    out : ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray or scalar
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`. This is a scalar if `x` is a scalar.
    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.sqrt, _np.sqrt, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def cbrt(x, out=None, **kwargs):
    r"""
    Return the cube-root of an array, element-wise.
    Parameters
    ----------
    x : ndarray
        The values whose cube-roots are required.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
    Returns
    ----------
    y : ndarray
        An array of the same shape as x, containing the cube cube-root of each element in x.
        If out was provided, y is a reference to it. This is a scalar if x is a scalar.
    Examples
    ----------
    >>> np.cbrt([1,8,27])
    array([ 1.,  2.,  3.])
    """
    return _unary_func_helper(x, _npi.cbrt, _np.cbrt, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def abs(x, out=None, **kwargs):
    r"""abs(x, out=None, **kwargs)
    Calculate the absolute value element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of
        each element in `x`. This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> x = np.array([-1.2, 1.2])
    >>> np.abs(x)
    array([1.2, 1.2])
    """
    return _unary_func_helper(x, _npi.abs, _np.abs, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def absolute(x, out=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.
    np.abs is a shorthand for this function.
    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
    Returns
    ----------
    absolute : ndarray
        An ndarray containing the absolute value of each element in x.
    Examples
    ----------
    >>> x = np.array([-1.2, 1.2])
    >>> np.absolute(x)
    array([ 1.2,  1.2])
    """
    return _unary_func_helper(x, _npi.absolute, _np.absolute, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def sign(x, out=None, **kwargs):
    r"""
    sign(x, out=None)
    Returns an element-wise indication of the sign of a number.
    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``. Only supports real number.
    Parameters
    ----------
    x : ndarray or a scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The sign of `x`.
        This is a scalar if `x` is a scalar.
    Note
    -------
    - Only supports real number as input elements.
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.
    Examples
    --------
    >>> a = np.array([-5., 4.5])
    >>> np.sign(a)
    array([-1.,  1.])
    Scalars as input:
    >>> np.sign(4.0)
    1.0
    >>> np.sign(0)
    0
    Use ``out`` parameter:
    >>> b = np.zeros((2, ))
    >>> np.sign(a, out=b)
    array([-1.,  1.])
    >>> b
    array([-1.,  1.])
    """
    return _unary_func_helper(x, _npi.sign, _np.sign, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def exp(x, out=None, **kwargs):
    r"""exp(x, out=None, **kwargs)
    Calculate the exponential of all elements in the input array.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> np.exp(1)
    2.718281828459045
    >>> x = np.array([-1, 1, -2, 2])
    >>> np.exp(x)
    array([0.36787945, 2.7182817 , 0.13533528, 7.389056  ])
    """
    return _unary_func_helper(x, _npi.exp, _np.exp, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def expm1(x, out=None, **kwargs):
    r"""expm1(x, out=None, **kwargs)
    Calculate `exp(x) - 1` of all elements in the input array.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential minus one: `out = exp(x) - 1`.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> np.expm1(1)
    1.718281828459045
    >>> x = np.array([-1, 1, -2, 2])
    >>> np.expm1(x)
    array([-0.63212056,  1.71828183, -0.86466472,  6.3890561])
    """
    return _unary_func_helper(x, _npi.expm1, _np.expm1, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def arcsin(x, out=None, **kwargs):
    r"""
    arcsin(x, out=None)
    Inverse sine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        `y`-coordinate on the unit circle.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    angle : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.
    Examples
    --------
    >>> np.arcsin(1)     # pi/2
    1.5707963267948966
    >>> np.arcsin(-1)    # -pi/2
    -1.5707963267948966
    >>> np.arcsin(0)
    0.0
    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].
    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    The inverse sine is also known as `asin` or sin^{-1}.
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.arcsin
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html>`_ in
    the following aspects:
    - Only support ndarray or scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.
    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/
    """
    return _unary_func_helper(x, _npi.arcsin, _np.arcsin, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def arccos(x, out=None, **kwargs):
    r"""
    Trigonometric inverse cosine, element-wise.
    The inverse of cos so that, if y = cos(x), then x = arccos(y).
    Parameters
    ----------
    x : ndarray
        x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
    Returns
    ----------
    angle : ndarray
        The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi].
        This is a scalar if x is a scalar.
    See also
    ----------
    cos, arctan, arcsin
    Notes
    ----------
    arccos is a multivalued function: for each x there are infinitely many numbers z such that
    cos(z) = x. The convention is to return the angle z whose real part lies in [0, pi].
    For real-valued input data types, arccos always returns real output.
    For each value that cannot be expressed as a real number or infinity, it yields nan and sets
    the invalid floating point error flag.
    The inverse cos is also known as acos or cos^-1.
    Examples
    ----------
    We expect the arccos of 1 to be 0, and of -1 to be pi:
    >>> np.arccos([1, -1])
    array([ 0.        ,  3.14159265])
    """
    return _unary_func_helper(x, _npi.arccos, _np.arccos, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def arctan(x, out=None, **kwargs):
    r"""arctan(x, out=None, **kwargs)
    Trigonometric inverse tangent, element-wise.
    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Out has the same shape as `x`. It lies is in
        ``[-pi/2, pi/2]`` (``arctan(+/-inf)`` returns ``+/-pi/2``).
        This is a scalar if `x` is a scalar.
    Notes
    -----
    `arctan` is a multi-valued function: for each `x` there are infinitely
    many numbers `z` such that tan(`z`) = `x`.  The convention is to return
    the angle `z` whose real part lies in [-pi/2, pi/2].
    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    For complex-valued input, we do not have support for them yet.
    The inverse tangent is also known as `atan` or tan^{-1}.
    Examples
    --------
    We expect the arctan of 0 to be 0, and of 1 to be pi/4:
    >>> x = np.array([0, 1])
    >>> np.arctan(x)
    array([0.       , 0.7853982])
    >>> np.pi/4
    0.7853981633974483
    """
    return _unary_func_helper(x, _npi.arctan, _np.arctan, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def log(x, out=None, **kwargs):
    """
    log(x, out=None)
    Natural logarithm, element-wise.
    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.
    Parameters
    ----------
    x : ndarray
        Input value. Elements must be of real value.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The natural logarithm of `x`, element-wise.
        This is a scalar if `x` is a scalar.
    Notes
    -----
     Currently only supports data of real values and ``inf`` as input. Returns data of real value, ``inf``, ``-inf`` and
    ``nan`` according to the input.
    This function differs from the original `numpy.log
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html>`_ in
    the following aspects:
    - Does not support complex number for now
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.
    Examples
    --------
    >>> a = np.array([1, np.exp(1), np.exp(2), 0], dtype=np.float64)
    >>> np.log(a)
    array([  0.,   1.,   2., -inf], dtype=float64)
    Due to internal calculation mechanism, using default float32 dtype may cause some special behavior:
    >>> a = np.array([1, np.exp(1), np.exp(2), 0], dtype=np.float32)
    >>> np.log(a)
    array([  0.,  0.99999994,   2., -inf])
    Scalar calculation:
    >>> np.log(1)
    0.0
    """
    return _unary_func_helper(x, _npi.log, _np.log, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def degrees(x, out=None, **kwargs):
    """
    degrees(x, out=None)
    Convert angles from radians to degrees.
    Parameters
    ----------
    x : ndarray
        Input value. Elements must be of real value.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The corresponding degree values; if `out` was supplied this is a
        reference to it.
        This is a scalar if `x` is a scalar.
    Notes
    -------
    This function differs from the original `numpy.degrees
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.degrees.html>`_ in
    the following aspects:
    - Input type does not support Python native iterables(list, tuple, ...). Only ndarray is supported.
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.
    Examples
    --------
    Convert a radian array to degrees
    >>> rad = np.arange(12.) * np.pi / 6
    >>> np.degrees(rad)
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    Use specified ``out`` ndarray:
    >>> out = np.zeros((rad.shape))
    >>> np.degrees(rad, out)
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    >>> out
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    """
    return _unary_func_helper(x, _npi.degrees, _np.degrees, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def rad2deg(x, out=None):
    r"""
    rad2deg(x, out=None)

    Convert angles from radians to degrees.
    Parameters
    ----------
    x : ndarray or scalar
        Angles in degrees.
    out : ndarray or None, optional
        A location into which the result is stored. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The corresponding angle in radians.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    "rad2deg(x)" is "x *180 / pi".

    This function differs from the original numpy.arange in the following aspects:
        - Only support float32 and float64.
        - `out` must be in the same size of input.

    Examples
    --------
    >>> np.rad2deg(np.pi/2)
    90.0
    """
    return _unary_func_helper(x, _npi.rad2deg, _np.rad2deg, out=out)


@set_module('mxnet.ndarray.numpy')
def rint(x, out=None, **kwargs):
    """
    Round elements of the array to the nearest integer.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
    Notes
    -----
    This function differs from the original `numpy.rint
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rint.html>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.rint(a)
    array([-2., -2., -0.,  0.,  1.,  2.,  2.])
    """
    return _unary_func_helper(x, _npi.rint, _np.rint, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def log2(x, out=None, **kwargs):
    """
    Base-2 logarithm of x.
    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The logarithm base two of `x`, element-wise.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    This function differs from the original `numpy.log2
    <https://www.google.com/search?q=numpy+log2>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param
    Examples
    --------
    >>> x = np.array([0, 1, 2, 2**4])
    >>> np.log2(x)
    array([-inf,   0.,   1.,   4.])
    """
    return _unary_func_helper(x, _npi.log2, _np.log2, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def log1p(x, out=None, **kwargs):
    """
    Return the natural logarithm of one plus the input array, element-wise.
    Calculates ``log(1 + x)``.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
        Natural logarithm of 1 + x, element-wise. This is a scalar
        if x is a scalar.
    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.
    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    cannot support complex-valued input.
    Examples
    --------
    >>> np.log1p(1e-99)
    1e-99
    >>> a = np.array([3, 4, 5])
    >>> np.log1p(a)
    array([1.3862944, 1.609438 , 1.7917595])
    """
    return _unary_func_helper(x, _npi.log1p, _np.log1p, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def radians(x, out=None, **kwargs):
    """
    Convert angles from degrees to radians.
    Parameters
    ----------
    x : ndarray or scalar
        Input array in degrees.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray
        The corresponding radian values. This is a scalar if x is a scalar.
    Notes
    -----
    This function differs from the original `numpy.radians
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.radians.html>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param
    Examples
    --------
    >>> deg = np.arange(12.) * 30.
    >>> np.radians(deg)
    array([0.       , 0.5235988, 1.0471976, 1.5707964, 2.0943952, 2.6179938,
           3.1415927, 3.6651914, 4.1887903, 4.712389 , 5.2359877, 5.7595863],
           dtype=float32)
    """
    return _unary_func_helper(x, _npi.radians, _np.radians, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def deg2rad(x, out=None):
    r"""
    deg2rad(x, out=None)

    Convert angles from degrees to radians.
    Parameters
    ----------
    x : ndarray or scalar
        Angles in degrees.
    out : ndarray or None, optional
        A location into which the result is stored. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The corresponding angle in radians.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    "deg2rad(x)" is "x * pi / 180".

    This function differs from the original numpy.arange in the following aspects:
        - Only support float32 and float64.
        - `out` must be in the same size of input.

    Examples
    --------
    >>> np.deg2rad(180)
    3.1415927
    """
    return _unary_func_helper(x, _npi.deg2rad, _np.deg2rad, out=out)


@set_module('mxnet.ndarray.numpy')
def reciprocal(x, out=None, **kwargs):
    r"""
    reciprocal(x, out=None)
    Return the reciprocal of the argument, element-wise.
    Calculates ``1/x``.
    Parameters
    ----------
    x : ndarray or scalar
        The values whose reciprocals are required.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
    Examples
    --------
    >>> np.reciprocal(2.)
    0.5
    >>> x = np.array([1, 2., 3.33])
    >>> np.reciprocal(x)
    array([1.       , 0.5      , 0.3003003])
    Notes
    -----
    .. note::
        This function is not designed to work with integers.
    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.reciprocal
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reciprocal.html>`_ in
    the following aspects:
    - Only support ndarray and scalar now.
    - `where` argument is not supported.
    """
    return _unary_func_helper(x, _npi.reciprocal, _np.reciprocal, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def square(x, out=None, **kwargs):
    r"""
    square(x, out=None)
    Return the element-wise square of the input.
    Parameters
    ----------
    x : ndarray or scalar
        The values whose squares are required.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.
    Returns
    -------
    y : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
    Examples
    --------
    >>> np.square(2.)
    4.0
    >>> x = np.array([1, 2., -1])
    >>> np.square(x)
    array([1., 4., 1.])
    Notes
    -----
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.square
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html>`_ in
    the following aspects:
    - Only support ndarray and scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.
    """
    return _unary_func_helper(x, _npi.square, _np.square, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def negative(x, out=None, where=True, **kwargs):
    r"""
    negative(x, out=None, where=True)
    Numerical negative, element-wise.
    Parameters:
    ------------
    x : ndarray or scalar
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
          A location into which the result is stored.
    where : ndarray, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone.
    Returns:
    ---------
    y : ndarray or scalar
        Returned array or scalar: y = -x. This is a scalar if x is a scalar.
    Examples:
    ---------
    >>> np.negative(1)
    -1
    """
    return _unary_func_helper(x, _npi.negative, _np.negative, out=out)


@set_module('mxnet.ndarray.numpy')
def fix(x, out=None):
    r"""
    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters:
    ----------
    x : ndarray
        An array of floats to be rounded
    out : ndarray, optional
        Output array
    Returns:
    -------
    y : ndarray of floats
    Examples
    ---------
    >>> np.fix(3.14)
    3
    """
    return _unary_func_helper(x, _npi.fix, _np.fix, out=out)


@set_module('mxnet.ndarray.numpy')
def tan(x, out=None, where=True, **kwargs):
    r"""
    tan(x, out=None, where=True)
    Compute tangent element-wise.
    Equivalent to np.sin(x)/np.cos(x) element-wise.

    Parameters:
    ----------
    x : ndarray
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
          A location into which the result is stored. If provided,
          it must have a shape that the inputs broadcast to. If not provided or None,
          a freshly-allocated array is returned. A tuple (possible only as a keyword argument)
          must have length equal to the number of outputs.
    where : ndarray, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone.
    Returns:
    -------
    y : ndarray
    The corresponding tangent values. This is a scalar if x is a scalar.
    Examples:
    >>> np.tan(0.5)
    0.5463024898437905
    """

    return _unary_func_helper(x, _npi.tan, _np.tan, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def ceil(x, out=None, **kwargs):
    r"""
    Return the ceiling of the input, element-wise.
    The ceil of the ndarray `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a same shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
        The ceiling of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])
    >>> #if you use parameter out, x and out must be ndarray. if not, you will get an error!
    >>> a = np.array(1)
    >>> np.ceil(np.array(3.5), a)
    array(4.)
    >>> a
    array(4.)
    """
    return _unary_func_helper(x, _npi.ceil, _np.ceil, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def floor(x, out=None, **kwargs):
    r"""
    Return the floor of the input, element-wise.
    The floor of the ndarray `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a same shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.
    Returns
    -------
    y : ndarray or scalar
        The floor of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.
    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])
    >>> #if you use parameter out, x and out must be ndarray. if not, you will get an error!
    >>> a = np.array(1)
    >>> np.floor(np.array(3.5), a)
    array(3.)
    >>> a
    array(3.)
    """
    return _unary_func_helper(x, _npi.floor, _np.floor, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def trunc(x, out=None, **kwargs):
    r"""
    trunc(x, out=None)
    Return the truncated value of the input, element-wise.
    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : ndarray or scalar
        Input data.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : ndarray or scalar
        The truncated value of each element in `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    This function differs from the original numpy.trunc in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.trunc(a)
    array([-1., -1., -0.,  0.,  1.,  1.,  2.])
    """
    return _unary_func_helper(x, _npi.trunc, _np.trunc, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def logical_not(x, out=None, **kwargs):
    r"""
    logical_not(x, out=None)
    Compute the truth value of NOT x element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Logical NOT is applied to the elements of `x`.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    This function differs from the original numpy.logical_not in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.
    Examples
    --------
    >>> x= np.array([True, False, 0, 1])
    >>> np.logical_not(x)
    array([0., 1., 1., 0.])

    >>> x = np.arange(5)
    >>> np.logical_not(x<3)
    array([0., 0., 0., 1., 1.])
    """
    return _unary_func_helper(x, _npi.logical_not, _np.logical_not, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def arcsinh(x, out=None, **kwargs):
    r"""
    arcsinh(x, out=None)
    Inverse hyperbolic sine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arcsinh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arcsinh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. DType of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.
    Examples
    --------
    >>> a = np.array([3.2, 5.0])
    >>> np.arcsinh(a)
    array([1.8309381, 2.2924316])
    >>> np.arcsinh(1)
    0.0
    """
    return _unary_func_helper(x, _npi.arcsinh, _np.arcsinh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def arccosh(x, out=None, **kwargs):
    r"""
    arccosh(x, out=None)
    Inverse hyperbolic cosine, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arccosh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arccosh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.
    Examples
    --------
    >>> a = np.array([3.2, 5.0])
    >>> np.arccosh(a)
    array([1.8309381, 2.2924316])
    >>> np.arccosh(1)
    0.0
    """
    return _unary_func_helper(x, _npi.arccosh, _np.arccosh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def arctanh(x, out=None, **kwargs):
    r"""
    arctanh(x, out=None)
    Inverse hyperbolic tangent, element-wise.
    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arctanh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.
    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arctanh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.
    Examples
    --------
    >>> a = np.array([0.0, -0.5])
    >>> np.arctanh(a)
    array([0., -0.54930615])
    >>> np.arctanh(0.0)
    0.0
    """
    return _unary_func_helper(x, _npi.arctanh, _np.arctanh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def tile(A, reps):
    r"""
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : ndarray or scalar
        An input array or a scalar to repeat.
    reps : a single integer or tuple of integers
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0., 1., 2., 0., 1., 2.])
    >>> np.tile(a, (2, 2))
    array([[0., 1., 2., 0., 1., 2.],
           [0., 1., 2., 0., 1., 2.]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0., 1., 2., 0., 1., 2.]],
           [[0., 1., 2., 0., 1., 2.]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1., 2., 1., 2.],
           [3., 4., 3., 4.]])
    >>> np.(b, (2, 1))
    array([[1., 2.],
           [3., 4.],
           [1., 2.],
           [3., 4.]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.]])

    Scalar as input:

    >>> np.tile(2, 3)
    array([2, 2, 2]) # repeating integer `2`

    """
    return _unary_func_helper(A, _npi.tile, _np.tile, reps=reps)

# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.
    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
          - ary[:2]
          - ary[2:3]
          - ary[3:]
        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.
    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.
    """
    indices = []
    axis_size = ary.shape[axis]
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if axis_size % sections:
            raise ValueError('array split does not result in an equal division')
        section_size = int(axis_size / sections)
        indices = [i * section_size for i in range(sections)]
    elif isinstance(indices_or_sections, tuple):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    ret = _npi.split(ary, indices, axis, False)
    if not isinstance(ret, list):
        return [ret]
    return ret
# pylint: enable=redefined-outer-name


@set_module('mxnet.ndarray.numpy')
def concatenate(seq, axis=0, out=None):
    """Join a sequence of arrays along an existing axis.
    Parameters
    ----------
    a1, a2, ... : sequence of ndarray
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    Returns
    -------
    res : ndarray
        The concatenated array.
    """
    return _npi.concatenate(*seq, dim=axis, out=out)


@set_module('mxnet.ndarray.numpy')
def stack(arrays, axis=0, out=None):
    """Join a sequence of arrays along a new axis.
        The axis parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.
    Parameters
    ----------
    arrays : sequence of ndarray
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what stack would have returned if no out argument were specified.
    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays."""
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.stack(*arrays, axis=axis, out=out)


@set_module('mxnet.ndarray.numpy')
def vstack(arrays, out=None):
    r"""Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate` and `stack`
    provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.vstack((a, b))
    array([[1., 2., 3.],
            [2., 3., 4.]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[2], [3], [4]])
    >>> np.vstack((a, b))
    array([[1.],
            [2.],
            [3.],
            [2.],
            [3.],
            [4.]])
    """
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.vstack(*arrays)


@set_module('mxnet.ndarray.numpy')
def maximum(x1, x2, out=None):
    """Returns element-wise maximum of the input arrays with broadcasting.

    Parameters
    ----------
    x1, x2 : scalar or mxnet.numpy.ndarray
        The arrays holding the elements to be compared. They must have the same shape,
        or shapes that can be broadcast to a single shape.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars."""
    return _ufunc_helper(x1, x2, _npi.maximum, _np.maximum, _npi.maximum_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def minimum(x1, x2, out=None):
    """Returns element-wise minimum of the input arrays with broadcasting.

    Parameters
    ----------
    x1, x2 : scalar or mxnet.numpy.ndarray
        The arrays holding the elements to be compared. They must have the same shape,
        or shapes that can be broadcast to a single shape.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        The minimum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars."""
    return _ufunc_helper(x1, x2, _npi.minimum, _np.minimum, _npi.minimum_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        Swapped array. This is always a copy of the input array.
    """
    return _npi.swapaxes(a, dim1=axis1, dim2=axis2)


@set_module('mxnet.ndarray.numpy')
def clip(a, a_min, a_max, out=None):
    """clip(a, a_min, a_max, out=None)

    Clip (limit) the values in an array.
    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : ndarray
        Array containing elements to clip.
    a_min : scalar or `None`
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    a_max : scalar or `None`
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Notes
    -----
    ndarray `a_min` and `a_max` are not supported.

    Examples
    --------
    >>> a = np.arange(10)
    >>> np.clip(a, 1, 8)
    array([1., 1., 2., 3., 4., 5., 6., 7., 8., 8.], dtype=float32)
    >>> a
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)
    >>> np.clip(a, 3, 6, out=a)
    array([3., 3., 3., 3., 4., 5., 6., 6., 6., 6.], dtype=float32)
    """
    if a_min is None and a_max is None:
        raise ValueError('array_clip: must set either max or min')
    if a_min is None:
        a_min = float('-inf')
    if a_max is None:
        a_max = float('inf')
    return _npi.clip(a, a_min, a_max, out=out)


@set_module('mxnet.ndarray.numpy')
def argmax(a, axis=None, out=None):
    r"""
    argmax(a, axis=None, out=None)

    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : ndarray
        Input array. Only support ndarrays of dtype `float16`, `float32`, and `float64`.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    index_array : ndarray of indices whose dtype is same as the input ndarray.
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    This function differs from the original `numpy.argmax
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - Output has dtype that is same as the input ndarray.
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10., 11., 12.],
           [13., 14., 15.]])
    >>> np.argmax(a)
    array(5.)
    >>> np.argmax(a, axis=0)
    array([1., 1., 1.])
    >>> np.argmax(a, axis=1)
    array([2., 2.])

    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0., 5., 2., 3., 4., 5.])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    array(1.)

    Specify ``out`` ndarray:

    >>> a = np.arange(6).reshape(2,3) + 10
    >>> b = np.zeros((2,))
    >>> np.argmax(a, axis=1, out=b)
    array([2., 2.])
    >>> b
    array([2., 2.])
    """
    return _npi.argmax(a, axis=axis, keepdims=False, out=out)


@set_module('mxnet.ndarray.numpy')
def mean(a, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
    """
    mean(a, axis=None, dtype=None, out=None, keepdims=None)
    Compute the arithmetic mean along the specified axis.
    Returns the average of the array elements.
    The average is taken over the flattened array by default, otherwise over the specified axis.
    Parameters
    ----------
    a : ndarray
        ndarray containing numbers whose mean is desired.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default is float32;
        for floating point inputs, it is the same as the input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result. The default is None; if provided,
        it must have the same shape and type as the expected output
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
        If the default value is passed, then keepdims will not be passed through to the mean
        method of sub-classes of ndarray, however any non-default value will be. If the sub-class
        method does not implement keepdims any exceptions will be raised.
    Returns
    -------
    m : ndarray, see dtype parameter above
        If out=None, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.
    Notes
    -----
    This function differs from the original `numpy.mean
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html>`_ in
    the following way(s):
    - only ndarray is accepted as valid input, python iterables or scalar is not supported
    - default data type for integer input is float32
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    array(2.5)
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0,:] = 1.0
    >>> a[1,:] = 0.1
    >>> np.mean(a)
    array(0.55)
    >>> np.mean(a, dtype=np.float64)
    array(0.55)
    """
    return _npi.mean(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@set_module('mxnet.ndarray.numpy')
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=too-many-arguments
    """
    Compute the standard deviation along the specified axis.
    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : ndarray
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949 # may vary
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])
    In single precision, std() can be inaccurate:
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.std(a)
    array(0.45)
    >>> np.std(a, dtype=np.float64)
    array(0.45, dtype=float64)
    """
    return _npi.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)


@set_module('mxnet.ndarray.numpy')
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=too-many-arguments
    """
    Compute the variance along the specified axis.
    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : ndarray
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float32`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    array(1.25)
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.var(a)
    array(0.2025)
    >>> np.var(a, dtype=np.float64)
    array(0.2025, dtype=float64)
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025
    """
    return _npi.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def indices(dimensions, dtype=_np.int32, ctx=None):
    """Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0,1,...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : data-type, optional
        The desired data-type for the array. Default is `float32`.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    grid : ndarray
        The array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

    Notes
    -----
    The output shape is obtained by prepending the number of dimensions
    in front of the tuple of dimensions, i.e. if `dimensions` is a tuple
    ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N,r0,...,rN-1)``.

    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k,i0,i1,...,iN-1] = ik

    Examples
    --------
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 0, 0],
           [1, 1, 1]], dtype=int32)

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0., 1., 2.],
           [4., 5., 6.]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.
    """
    if isinstance(dimensions, (tuple, list)):
        if ctx is None:
            ctx = current_context()
        return _npi.indices(dimensions=dimensions, dtype=dtype, ctx=ctx)
    else:
        raise ValueError("The dimensions must be sequence of ints")
# pylint: enable=redefined-outer-name


@set_module('mxnet.ndarray.numpy')
def copysign(x1, x2, out=None):
    r"""copysign(x1, x2, out=None)

    Change the sign of x1 to that of x2, element-wise.

    If `x2` is a scalar, its sign will be copied to all elements of `x1`.

    Parameters
    ----------
    x1 : ndarray or scalar
        Values to change the sign of.
    x2 : ndarray or scalar
        The sign of `x2` is copied to `x1`.
    out : ndarray or None, optional
        A location into which the result is stored. It must be of the
        right shape and right type to hold the output. If not provided
        or `None`,a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        The values of `x1` with the sign of `x2`.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -------
    This function differs from the original `numpy.copysign
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.copysign.html>`_ in
    the following aspects:

    - ``where`` param is not supported.

    Examples
    --------
    >>> np.copysign(1.3, -1)
    -1.3
    >>> 1/np.copysign(0, 1)
    inf
    >>> 1/np.copysign(0, -1)
    -inf

    >>> a = np.array([-1, 0, 1])
    >>> np.copysign(a, -1.1)
    array([-1., -0., -1.])
    >>> np.copysign(a, np.arange(3)-1)
    array([-1.,  0.,  1.])
    """
    return _ufunc_helper(x1, x2, _npi.copysign, _np.copysign, _npi.copysign_scalar, _npi.rcopysign_scalar, out)


@set_module('mxnet.ndarray.numpy')
def ravel(x, order='C'):
    r"""
    ravel(x)

    Return a contiguous flattened array.
    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    x : ndarray
        Input array.  The elements in `x` are read in row-major, C-style order and
        packed as a 1-D array.
    order : `C`, optional
        Only support row-major, C-style order.

    Returns
    -------
    y : ndarray
        y is an array of the same subtype as `x`, with shape ``(x.size,)``.
        Note that matrices are special cased for backward compatibility, if `x`
        is a matrix, then y is a 1-D ndarray.

    Notes
    -----
    This function differs from the original numpy.arange in the following aspects:
        - Only support row-major, C-style order.

    Examples
    --------
    It is equivalent to ``reshape(x, -1)``.

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print(np.ravel(x))
    [1. 2. 3. 4. 5. 6.]

    >>> print(x.reshape(-1))
    [1. 2. 3. 4. 5. 6.]

    >>> print(np.ravel(x.T))
    [1. 4. 2. 5. 3. 6.]
    """
    if order != 'C':
        raise NotImplementedError('order {} is not supported'.format(order))
    if isinstance(x, numeric_types):
        return _np.reshape(x, -1)
    elif isinstance(x, NDArray):
        return _npi.reshape(x, -1)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
def hanning(M, dtype=_np.float32, ctx=None):
    r"""Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `float32`. Note that you need
        select numpy.float32 or float64 in this operator.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    blackman, hamming

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hanning was named for Julius von Hann, an Austrian meteorologist.
    It is also known as the Cosine Bell. Some authors prefer that it be
    called a Hann window, to help avoid confusion with the very similar
    Hamming window.

    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> np.hanning(12)
    array([0.        , 0.07937324, 0.29229254, 0.5711574 , 0.8274304 ,
           0.9797465 , 0.97974646, 0.82743025, 0.5711573 , 0.29229245,
           0.07937312, 0.        ])

    Plot the window and its frequency response:

    >>> import matplotlib.pyplot as plt
    >>> window = np.hanning(51)
    >>> plt.plot(window.asnumpy())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("Hann window")
    Text(0.5, 1.0, 'Hann window')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("Sample")
    Text(0.5, 0, 'Sample')
    >>> plt.show()
    """
    if ctx is None:
        ctx = current_context()
    return _npi.hanning(M, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def hamming(M, dtype=_np.float32, ctx=None):
    r"""Return the hamming window.

    The hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `float32`. Note that you need
        select numpy.float32 or float64 in this operator.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    blackman, hanning

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
    and is described in Blackman and Tukey. It was recommended for
    smoothing the truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> np.hamming(12)
    array([0.08000001, 0.15302339, 0.34890914, 0.6054648 , 0.841236  ,
           0.9813669 , 0.9813668 , 0.8412359 , 0.6054647 , 0.34890908,
           0.15302327, 0.08000001])

    Plot the window and its frequency response:

    >>> import matplotlib.pyplot as plt
    >>> window = np.hamming(51)
    >>> plt.plot(window.asnumpy())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("hamming window")
    Text(0.5, 1.0, 'hamming window')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("Sample")
    Text(0.5, 0, 'Sample')
    >>> plt.show()
    """
    if ctx is None:
        ctx = current_context()
    return _npi.hamming(M, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def blackman(M, dtype=_np.float32, ctx=None):
    r"""Return the Blackman window.

    The Blackman window is a taper formed by using the first three
    terms of a summation of cosines. It was designed to have close to the
    minimal leakage possible.  It is close to optimal, only slightly worse
    than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `float32`. Note that you need
        select numpy.float32 or float64 in this operator.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    hamming, hanning

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/{M-1}) + 0.08 \cos(4\pi n/{M-1})

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the kaiser window.

    References
    ----------
    Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
    Dover Publications, New York.

    Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
    Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

    Examples
    --------
    >>> np.blackman(12)
    array([-1.4901161e-08,  3.2606423e-02,  1.5990365e-01,  4.1439798e-01,
            7.3604530e-01,  9.6704686e-01,  9.6704674e-01,  7.3604506e-01,
            4.1439781e-01,  1.5990359e-01,  3.2606363e-02, -1.4901161e-08])

    Plot the window and its frequency response:

    >>> import matplotlib.pyplot as plt
    >>> window = np.blackman(51)
    >>> plt.plot(window.asnumpy())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("blackman window")
    Text(0.5, 1.0, 'blackman window')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("Sample")
    Text(0.5, 0, 'Sample')
    >>> plt.show()
    """
    if ctx is None:
        ctx = current_context()
    return _npi.blackman(M, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def flip(m, axis=None, out=None):
    r"""
    flip(m, axis=None, out=None)

    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : ndarray or scalar
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to flip over. The default,
        axis=None, will flip over all of the axes of the input array.
        If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, flipping is performed on all of the axes
        specified in the tuple.
    out : ndarray or scalar, optional
        Alternative output array in which to place the result. It must have
        the same shape and type as the expected output.

    Returns
    -------
    out : ndarray or scalar
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> np.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    """
    from ...numpy import ndarray
    if isinstance(m, numeric_types):
        return _np.flip(m, axis)
    elif isinstance(m, ndarray):
        return _npi.flip(m, axis, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(m))))


@set_module('mxnet.ndarray.numpy')
def around(x, decimals=0, out=None, **kwargs):
    r"""
    around(x, decimals=0, out=None)

    Evenly round to the given number of decimals.
    Parameters
    ----------
    x : ndarray or scalar
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and type as the expected output.

    Returns
    -------
    rounded_array : ndarray or scalar
        An array of the same type as `x`, containing the rounded values.
        A reference to the result is returned.

    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc.

    This function differs from the original numpy.prod in the following aspects:

        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot support complex-valued number.

    Examples
    --------
    >>> np.around([0.37, 1.64])
    array([ 0.,  2.])
    >>> np.around([0.37, 1.64], decimals=1)
    array([ 0.4,  1.6])
    >>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
    array([ 0.,  2.,  2.,  4.,  4.])
    >>> np.around([1, 2, 3, 11], decimals=1) # ndarray of ints is returned
    array([ 1,  2,  3, 11])
    >>> np.around([1, 2, 3, 11], decimals=-1)
    array([ 0,  0,  0, 10])
    """
    from ...numpy import ndarray
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, ndarray):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
def arctan2(x1, x2, out=None):
    r"""
    arctan2(x1, x2, out=None)

    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that ``arctan2(x1, x2)`` is
    the signed angle in radians between the ray ending at the origin and
    passing through the point (1,0), and the ray ending at the origin and
    passing through the point (`x2`, `x1`).  (Note the role reversal: the
    "`y`-coordinate" is the first function parameter, the "`x`-coordinate"
    is the second.)  By IEEE convention, this function is defined for
    `x2` = +/-0 and for either or both of `x1` and `x2` = +/-inf (see
    Notes for specific values).

    This function is not defined for complex-valued arguments; for the
    so-called argument of complex values, use `angle`.

    Parameters
    ----------
    x1 : ndarray or scalar
        `y`-coordinates.
    x2 : ndarray or scalar
        `x`-coordinates. `x2` must be broadcastable to match the shape of
        `x1` or vice versa.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        Array of angles in radians, in the range ``[-pi, pi]``. This is a scalar if
        `x1` and `x2` are scalars.

    Notes
    -----
    *arctan2* is identical to the `atan2` function of the underlying
    C library.  The following special values are defined in the C
    standard: [1]_

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
        > 0   +/-inf +0 / +pi
        < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers, as are +inf
    and -inf.

    This function differs from the original numpy.arange in the following aspects:
        - Only support float16, float32 and float64.

    References
    ----------
    .. [1] ISO/IEC standard 9899:1999, "Programming language C."

    Examples
    --------
    Consider four points in different quadrants:

    >>> x = np.array([-1, +1, +1, -1])
    >>> y = np.array([-1, -1, +1, +1])
    >>> np.arctan2(y, x) * 180 / np.pi
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> x = np.array([1, -1])
    >>> y = np.array([0, 0])
    >>> np.arctan2(x, y)
    array([ 1.5707964, -1.5707964])
    """
    return _ufunc_helper(x1, x2, _npi.arctan2, _np.arctan2,
                         _npi.arctan2_scalar, _npi.rarctan2_scalar, out=out)


@set_module('mxnet.ndarray.numpy')
def hypot(x1, x2, out=None):
    r"""
    Given the "legs" of a right triangle, return its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.

    Parameters
    ----------
    x1, x2 : ndarray
        Leg of the triangle(s).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.

    Returns
    -------
    z : ndarray
        The hypotenuse of the triangle(s).
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    This function differs from the original numpy.arange in the following aspects:
        - Only support float16, float32 and float64.

    Examples
    --------
    >>> np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    Example showing broadcast of scalar_like argument:

    >>> np.hypot(3*np.ones((3, 3)), [4])
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])
    """
    return _ufunc_helper(x1, x2, _npi.hypot, _np.hypot, _npi.hypot_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def ldexp(x1, x2, out=None):
    """
    Returns x1 * 2**x2, element-wise.
    The mantissas `x1` and twos exponents `x2` are used to construct
    floating point numbers ``x1 * 2**x2``.

    Parameters
    ----------
    x1 : ndarray or scalar
        Array of multipliers.
    x2 : ndarray or scalar, int
        Array of twos exponents.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The result of ``x1 * 2**x2``.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.
    Different from numpy, we allow x2 to be float besides int.
    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.

    Examples
    --------
    >>> np.ldexp(5, np.arange(4))
    array([  5.,  10.,  20.,  40.])
    """
    return _ufunc_helper(x1, x2, _npi.ldexp, _np.ldexp, _npi.ldexp_scalar, _npi.rldexp_scalar, out)


@set_module('mxnet.ndarray.numpy')
def inner(a, b):
    r"""
    Inner product of two arrays.
    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : ndarray
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : ndarray
        `out.shape = a.shape[:-1] + b.shape[:-1]`

    Raises
    ------
    ValueError
        If the last dimension of `a` and `b` has different size.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    einsum : Einstein summation convention.

    Notes
    -----
    For vectors (1-D arrays) it computes the ordinary inner-product::
        np.inner(a, b) = sum(a[:]*b[:])
    More generally, if `ndim(a) = r > 0` and `ndim(b) = s > 0`::
        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))
    or explicitly::
        np.inner(a, b)[i0,...,ir-1,j0,...,js-1]
            = sum(a[i0,...,ir-1,:]*b[j0,...,js-1,:])
    In addition `a` or `b` may be scalars, in which case::
    np.inner(a,b) = a*b

    Examples
    --------
    Ordinary inner product for vectors:
    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2
    A multidimensional example:
    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> np.inner(a, b)
    array([[ 14,  38,  62],
           [ 86, 110, 134]])
    """
    return tensordot(a, b, [-1, -1])


@set_module('mxnet.ndarray.numpy')
def outer(a, b):
    r"""
    Compute the outer product of two vectors.
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    [[a0*b0  a0*b1 ... a0*bN ]
    [a1*b0    .
    [ ...          .
    [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) ndarray
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) ndarray
        Second input vector.  Input is flattened if
        not already 1-dimensional.

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to N dimensions and other operations.
                ``np.multiply.outer(a.ravel(), b.ravel())`` is the equivalent.
    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
            ed., Baltimore, MD, Johns Hopkins University Press, 1996,
            pg. 8.
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.]])
    """
    return tensordot(a.flatten(), b.flatten(), 0)


@set_module('mxnet.ndarray.numpy')
def vdot(a, b):
    r"""
    Return the dot product of two vectors.
    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : ndarray
        First argument to the dot product.
    b : ndarray
        Second argument to the dot product.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
        first argument.

    Examples
    --------
    Note that higher-dimensional arrays are flattened!
    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.vdot(a, b)
    30
    >>> np.vdot(b, a)
    30
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30
    """
    return tensordot(a.flatten(), b.flatten(), 1)


@set_module('mxnet.ndarray.numpy')
def equal(x1, x2, out=None):
    """
    Return (x1 == x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    Examples
    --------
    >>> np.equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[False, False, False],
           [False, False, False]])
    >>> np.equal(1, np.ones(1))
    array([ True])
    """
    return _ufunc_helper(x1, x2, _npi.equal, _np.equal, _npi.equal_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def not_equal(x1, x2, out=None):
    """
    Return (x1 != x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.not_equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.not_equal(1, np.ones(1))
    array([False])
    """
    return _ufunc_helper(x1, x2, _npi.not_equal, _np.not_equal, _npi.not_equal_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def greater(x1, x2, out=None):
    """
    Return the truth value of (x1 > x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.greater(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.greater(1, np.ones(1))
    array([False])
    """
    return _ufunc_helper(x1, x2, _npi.greater, _np.greater, _npi.greater_scalar,
                         _npi.less_scalar, out)


@set_module('mxnet.ndarray.numpy')
def less(x1, x2, out=None):
    """
    Return the truth value of (x1 < x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.less(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.less(1, np.ones(1))
    array([False])
    """
    return _ufunc_helper(x1, x2, _npi.less, _np.less, _npi.less_scalar, _npi.greater_scalar, out)


@set_module('mxnet.ndarray.numpy')
def greater_equal(x1, x2, out=None):
    """
    Return the truth value of (x1 >= x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.greater_equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.greater_equal(1, np.ones(1))
    array([True])
    """
    return _ufunc_helper(x1, x2, _npi.greater_equal, _np.greater_equal, _npi.greater_equal_scalar,
                         _npi.less_equal_scalar, out)


@set_module('mxnet.ndarray.numpy')
def less_equal(x1, x2, out=None):
    """
    Return the truth value of (x1 <= x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.less_equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[False, False, False],
           [False, False, False]])
    >>> np.less_equal(1, np.ones(1))
    array([True])
    """
    return _ufunc_helper(x1, x2, _npi.less_equal, _np.less_equal, _npi.less_equal_scalar,
                         _npi.greater_equal_scalar, out)
