"""Check exact state restoration, transition continuity and observed start mixing."""
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from craftax.craftax_env import make_craftax_env_from_name
from recovery_resets import RecoveryResetWrapper
from symbolic_map_memory import with_map_memory


class RecoveryResetsTest(unittest.TestCase):
    def test_rewind_and_mix(self):
        env=with_map_memory(make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',False),
                            {'SYMBOLIC_ENCODER':'map_mlp'})
        key=jax.random.PRNGKey(6)
        obs,state=env.reset(key,env.default_params)
        for i in range(4):
            obs,state,*_=env.step(jax.random.fold_in(key,i),state,1,env.default_params)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bank'
            path.write_bytes(serialization.to_bytes(jax.tree_util.tree_map(lambda x:np.asarray(x)[None],(obs,state))))
            wrap=RecoveryResetWrapper(env,path,1.)
            restored,slot=wrap.reset(key,env.default_params)
            np.testing.assert_array_equal(restored,obs)
            for a,b in zip(jax.tree_util.tree_leaves(slot.env_state),jax.tree_util.tree_leaves(state)):
                np.testing.assert_array_equal(a,b)
            new,slot2,r,d,info=wrap.step(key,slot,3,env.default_params)
            expected,_,r2,d2,_=env.step(key,state,3,env.default_params)
            np.testing.assert_array_equal(new,expected)
            self.assertEqual(r,r2);self.assertEqual(d,d2)
            self.assertTrue(info['practice/recovery_start']);self.assertFalse(slot2.first)
            mix=RecoveryResetWrapper(env,path,.25)
            _,slots=jax.vmap(lambda k:mix.reset(k,env.default_params))(jax.random.split(key,128))
            self.assertGreater(int(slots.recovery.sum()),15);self.assertLess(int(slots.recovery.sum()),50)
            np.testing.assert_array_equal(slots.env_state.elapsed,4*slots.recovery)


if __name__=='__main__':unittest.main()
