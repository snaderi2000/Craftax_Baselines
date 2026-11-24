"""Fixed-world resets reproduce complete original simulator states."""
import json,tempfile,unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from fixed_world_seeds import FixedWorldSeedWrapper,read_world_seeds
from wrappers import ClassicDeathPenaltyWrapper


class FixedWorldSeedTests(unittest.TestCase):
    def test_bank_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'seeds.json'
            for values in ([],[1,1],[True],[2**32],[-1],[1.5]):
                path.write_text(json.dumps({'world_seeds':values}))
                with self.assertRaises(ValueError):read_world_seeds(path)
            path.write_text(json.dumps({'world_seeds':[123,456]}))
            self.assertEqual(read_world_seeds(path),[123,456])

    def test_exact_world_reproduction_and_unchanged_steps(self):
        env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
        shaped=ClassicDeathPenaltyWrapper(env,2.)
        seeds=[123,456];fixed=FixedWorldSeedWrapper(shaped,seeds)
        expected=[env.reset(jax.random.PRNGKey(s),env.default_params) for s in seeds]
        used=set()
        for index in range(8):
            key=jax.random.PRNGKey(index)
            chosen=int(jax.random.randint(key,(),0,2));used.add(chosen)
            actual=fixed.reset(key,env.default_params)
            for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected[chosen])):
                np.testing.assert_array_equal(a,b)
        self.assertEqual(used,{0,1})
        key=jax.random.PRNGKey(99)
        actual=fixed.step(key,actual[1],0,env.default_params)
        ordinary=shaped.step(key,expected[chosen][1],0,env.default_params)
        for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(ordinary)):
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
