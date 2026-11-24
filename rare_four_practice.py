"""Normal-episode rarity selects four bounded-reuse practice banks."""
from functools import partial
import jax
import jax.numpy as jnp
from id_practice import IDPractice


class RareFourPractice(IDPractice):
    def __init__(self,base,num_envs=128,practice_envs=13,capacity=64,horizon=64,
                 window=512,refresh=128,min_episodes=512,max_reuse=10,full_episode=False):
        super().__init__(base,num_envs,practice_envs,capacity,horizon,full_episode)
        self.rare_four = True
        self.window,self.refresh,self.min_episodes,self.max_reuse = window,refresh,min_episodes,max_reuse

    @partial(jax.jit,static_argnums=(0,2))
    def reset(self,key,params=None):
        obs,state = super().reset(key,params)
        state = state.replace(rarity_window=jnp.zeros((self.window,22),jnp.int32),
            rarity_sum=jnp.zeros(22,jnp.int32),rarity_total=jnp.asarray(0),rarity_last_update=jnp.asarray(0),
            rarity_selected=jnp.zeros(22,bool),rarity_selection_counts=jnp.zeros(22,jnp.int32),
            reuse_counts=jnp.zeros((22,self.capacity),jnp.int32),rarity_starts=jnp.zeros(22,jnp.int32),
            rarity_unique_starts=jnp.zeros(22,jnp.int32))
        return obs,state

    def _record(self,state,next_raw):
        before = state
        state = super()._record(state,next_raw)
        changed = ((state.archive_event_steps != before.archive_event_steps)
                   | (state.archive_workers != before.archive_workers)
                   | (state.archive_episodes != before.archive_episodes)
                   | (state.archive_lags != before.archive_lags))
        return state.replace(reuse_counts=jnp.where(changed,0,state.reuse_counts))

    def _usable(self,state):
        return (jnp.arange(self.capacity)[None,:] < state.archive_counts[:,None]) & (state.reuse_counts < self.max_reuse)

    def _update_rarity(self,state,next_raw,normal_done,key):
        def record(i,carry):
            def insert(carry):
                window,sums,total = carry
                index = total%self.window
                flags = next_raw.achievements[i].astype(jnp.int32)
                sums = sums-window[index]+flags
                return window.at[index].set(flags),sums,total+1
            return jax.lax.cond(normal_done[i],insert,lambda c:c,carry)
        window,sums,total = jax.lax.fori_loop(0,self.num_envs,record,
            (state.rarity_window,state.rarity_sum,state.rarity_total))
        state = state.replace(rarity_window=window,rarity_sum=sums,rarity_total=total)
        refresh = (total>=self.min_episodes) & (total-state.rarity_last_update>=self.refresh)
        def select(state):
            rates = state.rarity_sum/jnp.maximum(1,jnp.minimum(state.rarity_total,self.window))
            jitter = jax.random.uniform(jax.random.fold_in(key,911),(22,))*1e-6
            scores = jnp.where(self._usable(state).any(axis=1),-rates+jitter,-jnp.inf)
            values,ids = jax.lax.top_k(scores,4)
            selected = jnp.zeros(22,bool).at[ids].set(jnp.isfinite(values))
            return state.replace(rarity_selected=selected,rarity_last_update=total,
                rarity_selection_counts=state.rarity_selection_counts+selected.astype(jnp.int32))
        return jax.lax.cond(refresh,select,lambda s:s,state)

    def _reset_choices(self,key,normal,state,done):
        targets = jnp.full(self.num_envs,-1,jnp.int32)
        def lane(i,carry):
            starts,targets,state = carry
            def select(carry):
                starts,targets,state = carry
                usable = self._usable(state)
                available = state.rarity_selected & usable.any(axis=1)
                def use(carry):
                    starts,targets,state = carry
                    tk,ek = jax.random.split(jax.random.fold_in(key,i))
                    task = jax.random.categorical(tk,jnp.where(available,0.,-1e9)).astype(jnp.int32)
                    entry = jax.random.categorical(ek,jnp.where(usable[task],0.,-1e9)).astype(jnp.int32)
                    starts = jax.tree_util.tree_map(lambda x,a:x.at[i].set(a[task,entry]),starts,state.archive)
                    first = state.reuse_counts[task,entry]==0
                    state = state.replace(reuse_counts=state.reuse_counts.at[task,entry].add(1),
                        rarity_starts=state.rarity_starts.at[task].add(1),
                        rarity_unique_starts=state.rarity_unique_starts.at[task].add(first.astype(jnp.int32)))
                    return starts,targets.at[i].set(task),state
                return jax.lax.cond(available.any(),use,lambda c:c,carry)
            # Sequentially consume starts: simultaneous resets cannot exceed the cap.
            return jax.lax.cond(done[i],select,lambda c:c,carry)
        return jax.lax.fori_loop(self.num_envs-self.practice_envs,self.num_envs,lane,(normal,targets,state))
