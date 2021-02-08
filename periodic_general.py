'''
Updated version for jax_md.space.periodic_general()
https://gist.github.com/sschoenholz/14944c4b9dd263c95c524f84cc1c4287#file-periodic_general-py
'''

from jax_md import space
from jax import custom_jvp
from jax import lax
import jax.numpy as np

periodic_displacement = space.periodic_displacement
pairwise_displacement = space.pairwise_displacement
periodic_shift = space.periodic_shift

f32 = np.float32

def inverse(box):
  if np.isscalar(box) or box.size == 1:
    return 1 / box
  elif box.ndim == 1:
    return 1 / box
  elif box.ndim == 2:
    return np.linalg.inv(box)
  
  raise ValueError()

def get_free_indices(n):
  return ''.join([chr(ord('a') + i) for i in range(n)])

@custom_jvp
def transform(box, R):
  if np.isscalar(box) or box.size == 1:
    return R * box
  elif box.ndim == 1:
    indices = get_free_indices(R.ndim - 1) + 'i'
    return np.einsum(f'i,{indices}->{indices}', box, R)
  elif box.ndim == 2:
    free_indices = get_free_indices(R.ndim - 1)
    left_indices = free_indices + 'j'
    right_indices = free_indices + 'i'
    return np.einsum(f'ij,{left_indices}->{right_indices}', box, R)
  raise ValueError()

@transform.defjvp
def transform_jvp(primals, tangents):
  box, R = primals
  dbox, dR = tangents

  return transform(box, R), dR + transform(dbox, R)

def periodic_general(box, wrapped=True):

  inv_box = inverse(box)

  def displacement_fn(Ra, Rb, **kwargs):
    _box, _inv_box = box, inv_box

    if 'box' in kwargs:      
      _box = kwargs['box']

    dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
    return transform(_box, dR) 

  def u(R, dR):
    if wrapped:
      return periodic_shift(f32(1.0), R, dR)
    return R + dR

  def shift_fn(R, dR, **kwargs):
    _box, _inv_box = box, inv_box
    if 'box' in kwargs:
      _box = kwargs['box']
      _inv_box = inverse(_box)
    dR = transform(_inv_box, dR)
    R = u(R, dR)
    return R
  
  return displacement_fn, shift_fn