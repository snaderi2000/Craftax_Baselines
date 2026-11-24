"""Per-ID mastery-gated backward practice with four offset-specific banks."""
from functools import partial
import jax
import jax.numpy as jnp
from rare_four_practice import RareFourPractice

LEVEL_OFFSETS = (1,5,20,50)


class BackwardPractice(RareFourPractice):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.backward = True
        self.offsets = LEVEL_OFFSETS
        self.history_length = 50
        self.per_level_capacity = self.capacity//4

    @partial(jax.jit,static_argnums=(0,2))
    def reset(self,key,params=None):
        obs,state = super().reset(key,params)
        return obs,state.replace(backward_bank_counts=jnp.zeros((22,4),jnp.int32),
            backward_bank_totals=jnp.zeros((22,4),jnp.int32),backward_level=jnp.zeros(22,jnp.int32),
            backward_outcomes=jnp.zeros((22,4,128),jnp.int32),backward_attempt_counts=jnp.zeros((22,4),jnp.int32),
            backward_last_check=jnp.zeros((22,4),jnp.int32),backward_promotions=jnp.zeros(22,jnp.int32),
            practice_level=jnp.full(self.num_envs,-1,jnp.int32))

    def _record(self,state,next_raw):
        history = jax.tree_util.tree_map(lambda h,x:h.at[state.cursor].set(x),state.history,state.raw)
        indices = (state.cursor-jnp.asarray(self.offsets)+1)%50
        candidates = jax.tree_util.tree_map(lambda h:jnp.take(h,indices,axis=0),history)
        new_events = next_raw.achievements & ~state.raw.achievements
        valid = (state.history_count[None,:]+1>=jnp.asarray(self.offsets)[:,None]) & (state.target[None,:]<0)
        k = min(self.per_level_capacity,self.num_envs)
        state = state.replace(history=history)
        def bank(b,state):
            task,level = b//4,b%4
            match = valid[level] & new_events[:,task] & ~candidates.achievements[level,:,task]
            def insert(state):
                ranks,ix = jax.lax.top_k(jnp.where(match,jnp.arange(self.num_envs),-1),k)
                good = ranks>=0
                slots = jnp.where(good,level*self.per_level_capacity+
                    (state.backward_bank_totals[task,level]+jnp.arange(k))%self.per_level_capacity,self.capacity)
                archive = jax.tree_util.tree_map(lambda a,x:a.at[task].set(
                    a[task].at[slots].set(x[level,ix],mode='drop')),state.archive,candidates)
                def put(table,values):
                    return table.at[task].set(table[task].at[slots].set(values,mode='drop'))
                n = good.sum()
                counts = state.backward_bank_counts.at[task,level].set(
                    jnp.minimum(self.per_level_capacity,state.backward_bank_counts[task,level]+n))
                totals = state.backward_bank_totals.at[task,level].add(n)
                return state.replace(archive=archive,archive_counts=counts.sum(axis=1),archive_totals=totals.sum(axis=1),
                    backward_bank_counts=counts,backward_bank_totals=totals,
                    archive_lags=put(state.archive_lags,jnp.full(k,jnp.asarray(self.offsets)[level])),
                    archive_workers=put(state.archive_workers,ix),
                    archive_episodes=put(state.archive_episodes,state.episode_serial[ix]),
                    archive_event_steps=put(state.archive_event_steps,next_raw.timestep[ix]),
                    reuse_counts=put(state.reuse_counts,jnp.zeros(k,jnp.int32)))
            return jax.lax.cond(match.any(),insert,lambda s:s,state)
        return jax.lax.fori_loop(0,88,bank,state)

    def _all_usable(self,state):
        valid = (jnp.arange(self.per_level_capacity)[None,None,:] < state.backward_bank_counts[:,:,None]).reshape(22,-1)
        return valid & (state.reuse_counts<self.max_reuse)

    def _usable(self,state):
        levels = jnp.arange(self.capacity)//self.per_level_capacity
        return self._all_usable(state) & (levels[None,:]==state.backward_level[:,None])

    def _reset_choices(self,key,normal,state,done):
        starts,targets,state = super()._reset_choices(key,normal,state,done)
        chosen_level = state.backward_level[jnp.maximum(targets,0)]
        return starts,targets,state.replace(practice_level=jnp.where(done,
            jnp.where(targets>=0,chosen_level,-1),state.practice_level))

    def _time_limit(self,state):
        return jnp.where(state.practice_level==3,128,self.horizon)

    def _update_curriculum(self,state,ended,success):
        def lane(i,state):
            def update(state):
                task,level = state.target[i],state.practice_level[i]
                n = state.backward_attempt_counts[task,level]
                outcomes = state.backward_outcomes.at[task,level,n%128].set(success[i].astype(jnp.int32))
                counts = state.backward_attempt_counts.at[task,level].add(1)
                current = state.backward_level[task]
                check = (level==current) & (n+1>=128) & (n+1-state.backward_last_check[task,level]>=32)
                rate = outcomes[task,level].mean()
                next_level = jnp.minimum(level+1,3)
                usable = self._all_usable(state).reshape(22,4,self.per_level_capacity)[task,next_level].any()
                promote = check & (rate>=.8) & (level<3) & usable
                return state.replace(backward_outcomes=outcomes,backward_attempt_counts=counts,
                    backward_last_check=state.backward_last_check.at[task,level].set(jnp.where(check,n+1,state.backward_last_check[task,level])),
                    backward_level=state.backward_level.at[task].add(promote.astype(jnp.int32)),
                    backward_promotions=state.backward_promotions.at[task].add(promote.astype(jnp.int32)))
            return jax.lax.cond(ended[i],update,lambda s:s,state)
        return jax.lax.fori_loop(0,self.num_envs,lane,state)
