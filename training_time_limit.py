"""Host-clock limit checked at PPO update boundaries inside compiled training."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback


def run_update_loop(update, runner_state, updates, deadline=0.):
    if not deadline:
        return jax.lax.scan(update,runner_state,None,updates)
    if updates<1:raise ValueError('A capped run requires at least one remaining update')
    # One completed update supplies the metric structure. The deadline includes
    # compilation; checks every 16 updates keep callback overhead small.
    first=update(runner_state,None)
    def condition(carry):
        state,_=carry
        def check(_):
            return io_callback(lambda:np.asarray(time.time()<deadline,np.bool_),
                jax.ShapeDtypeStruct((),jnp.bool_))
        before_deadline=jax.lax.cond(state[-1]%16==0,check,lambda _:jnp.asarray(True),None)
        return (state[-1]<updates)&before_deadline
    return jax.lax.while_loop(condition,lambda carry:update(carry[0],None),first)
