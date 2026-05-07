import jax.numpy as jnp
import numpy as np
from gibss2.cox import prepare_fixed_cox_context

def test_cox_context_sorting():
    # Times: 3.0 (event), 1.0 (censor), 2.0 (event), 2.0 (event)
    time = jnp.array([3.0, 1.0, 2.0, 2.0])
    event = jnp.array([1.0, 0.0, 1.0, 1.0])
    y = jnp.column_stack([time, event])
    
    ctx = prepare_fixed_cox_context(y)
    
    # Expected order: index 1 (time 1.0), 2 (2.0), 3 (2.0), 0 (3.0)
    np.testing.assert_array_equal(ctx.order, np.array([1, 2, 3, 0]))
    # Event group starts: index 1 and 3 in sorted array (times 2.0 and 3.0 have events)
    # Time 1.0 (index 0) is a censor, so it's not an event start.
    np.testing.assert_array_equal(ctx.event_group_starts, np.array([1, 3]))
    # Event counts: 2 events at time 2.0, 1 at time 3.0
    np.testing.assert_array_equal(ctx.event_counts, np.array([2.0, 1.0]))
