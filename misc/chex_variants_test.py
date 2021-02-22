import chex
from jax import jit
from unittest import skip

class ExampleTest(chex.TestCase):

  @skip
  def test_without_jit(self):
    print("\nWithout jit, the function is traced on every call (3 times).")

    def fn(x, y):
      print("Tracing fn")
      return x + y

    self.assertEqual(fn(1, 2), 3)
    self.assertEqual(fn(3, 4), 7)
    self.assertEqual(fn(5, 6), 11)

  @skip
  def test_with_jit(self):
    print("\nA jitted function will be traced only on the first call (once).")

    @jit
    def fn(x, y):
      print("Tracing fn")
      return x + y

    self.assertEqual(fn(1, 2), 3)
    self.assertEqual(fn(3, 4), 7)
    self.assertEqual(fn(5, 6), 11)


  @skip
  @chex.variants(with_jit=True, without_jit=True)
  def test_variant_not_prejitted(self):
    print("\nA variant of a non-jitted function will act like the previous two combined (traced 4 times")

    def fn(x, y):
      print("Tracing fn")
      return x + y

    var_fn = self.variant(fn)
    self.assertEqual(var_fn(1, 2), 3)
    self.assertEqual(var_fn(3, 4), 7)
    self.assertEqual(var_fn(5, 6), 11)
     

  @chex.variants(with_jit=True, without_jit=True)
  def test_variant_not_prejitted(self):
    print("\nA variant of a non-jitted function will act like the previous two combined (traced 4 times")

    @jit
    def fn(x, y):
      print("Tracing fn")
      return x + y

    var_fn = self.variant(fn)
    self.assertEqual(var_fn(1, 2), 3)
    self.assertEqual(var_fn(3, 4), 7)
    self.assertEqual(var_fn(5, 6), 11)
   