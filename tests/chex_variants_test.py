import chex

def fn(x, y):
  return x + y


# class ExampleTest(chex.TestCase):
# 
#   @chex.variants(with_jit=True, without_jit=True)
#   def test(self):
#     var_fn = self.variant(fn)
#     self.assertEqual(fn(1, 2), 3)
#     self.assertEqual(var_fn(1, 2), fn(1, 2))