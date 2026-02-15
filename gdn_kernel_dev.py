import tvm
from tvm import DataType, te, tir, s_tir, topi

def _gen_causal_conv1d_update(
        hidden_size: int, 
        state_len: int, 
        dtype: str = "float32", 
        activation: str ="silu",
        has_bias: bool =True
):
    _seq_len = tir.Var("seq_len", dtype="int64") # changes at runtime depending on prefill or decode
    _batch_size = tir.Var("batch_size", dtype="int64") # changes at runtime
    _hidden_size = tir.IntImm(dtype="int64", value=hidden_size) # from config
    _state_len = tir.IntImm(dtype="int64", value=state_len) # from config
    _dtype = DataType(dtype) # static
    hidden_states = te.placeholder(
            (_batch_size, _hidden_size, _seq_len), 
            dtype=_dtype, 
            name="hidden_states"
    )
    conv_state = te.placeholder(
            (_batch_size, _hidden_size, _state_len), 
            dtype=_dtype, 
            name="conv_state"
    )
    weight = te.placeholder(
            (_hidden_size, 1, _state_len), 
            dtype=_dtype, 
            name="weight"
    )
    bias = None
    if has_bias:
        bias = te.placeholder((_hidden_size,), dtype=_dtype, name="bias") 

    # Concatenation across the last dim
    total_len = _state_len + _seq_len
    hidden_states_new = te.compute(
        (_batch_size, _hidden_size, total_len),
        lambda b, h, l: te.if_then_else(
            l < _state_len, conv_state[b, h, l], hidden_states[b, h, l - _state_len]
        ),
        name="hidden_states_new"
    )

    # Create new conv state only last state_len on last dim
    next_conv_state = te.compute(
        (_batch_size, _hidden_size, _state_len),
        lambda b, h, s: hidden_states_new[b, h, s + _seq_len],
        name="next_conv_state"
    )

    # Compute the causal convolution across the last dim of new_hidden_states
    kw = te.reduce_axis((0, _state_len), name="kw")
    # conv_sum last dim is iterating across the input sequence (_seq_len)
    # cs[b,h,0] looks at hidden_states_new[b,h,0] -> hidden_states_new[b,h, state_len]
    # then, the next token slides the window one to the right, so it uses 
    # hidden_states_new[b,h,1] -> hidden_states_new[b,h,state_len+1]
    # which includes token 0's input hidden state, and excludes the oldest cached hidden state 
    conv_sum = te.compute(
        (_batch_size, _hidden_size, _seq_len),
        lambda b, h, s: te.sum(hidden_states_new[b, h, s + kw + 1] * weight[h, 0, kw], axis=kw),
        name="conv_sum"
    )

    def bias_and_act_logic(b, h, s):
        val = conv_sum[b, h, s]
        if has_bias:
            val += bias[h]
        if activation == "silu":
            val = val * te.sigmoid(val)
        return val

    out = te.compute((_batch_size, _hidden_size, _seq_len), bias_and_act_logic, name="out")

    # actually create primfunc 
    inputs = [hidden_states, conv_state, weight]
    if has_bias:
        inputs.append(bias)
    
    # return both, next_conv needed for cache update
    outputs = [next_conv_state, out]
    
    return te.create_prim_func(inputs + outputs)

def _chunk_gated_delta_rule(
        linear_num_value_heads: int, 
        linear_value_head_dim: int,
        linear_num_key_heads: int,
        linear_key_head_dim: int,
        chunk_size: int = 64,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        dtype: str = "float32"
        ):
    _linear_num_value_heads = tir.IntImm("int64", linear_num_value_heads)
    _linear_value_head_dim = tir.IntImm("int64", linear_value_head_dim)
    _linear_num_key_heads =  tir.IntImm("int64", linear_num_key_heads)
    _linear_key_head_dim =  tir.IntImm("int64", linear_key_head_dim)
    _chunk_size =  tir.IntImm("int64", chunk_size)
    _seq_len = tir.Var("seq_len", dtype="int64") # changes at runtime
    _batch_size = tir.Var("batch_size", dtype="int64") # changes at runtime
    _dtype = DataType(dtype)

    nodes = []

    query: te.Tensor = te.placeholder(
        shape=(_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
        dtype = _dtype,
        name="query"
    )
    
    key: te.Tensor = te.placeholder(
        shape=(_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
        dtype = _dtype,
        name="key"
    )
   
    value: te.Tensor = te.placeholder(
        shape=(_batch_size, _seq_len, _linear_num_value_heads, _linear_value_head_dim),
        dtype = _dtype,
        name="value"
    )

    g: te.Tensor = te.placeholder(
            (_batch_size, _seq_len, _linear_num_value_heads)
    )

    beta: te.Tensor = te.placeholder(
            (_batch_size, _seq_len, _linear_num_value_heads),
            dtype = _dtype,
            name="beta"
    )

    nodes.extend([query, key, value, g, beta])
    if use_qk_l2norm_in_kernel:
        # do l2 norm on query and key here
        query_sq = te.compute(
            query.shape, lambda b,s,n,d: query[b,s,n,d] * query[b,s,n,d], name="query_sq"
        )
        key_sq = te.compute(
            key.shape, lambda b,s,n,d: key[b,s,n,d] * key[b,s,n,d], name="key_sq"
        )

        rv = te.reduce_axis((0, _linear_key_head_dim), name="rv")
        query_sum = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, 1),
            lambda i,j,k,l: te.sum(query_sq[i,j,k,rv], axis=rv),
            name="query_sum"
        )
        key_sum = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, 1),
            lambda i,j,k,l: te.sum(key_sq[i,j,k,rv], axis=rv),
            name="key_sum"
        )
        eps = tir.FloatImm("float32", 1e-6) #hardcode for now
        q_norm = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
            lambda i,j,k,l: query[i,j,k,l] * te.rsqrt(query_sum[i,j,k,0] + eps),
            "q_norm"
        )
        k_norm = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
            lambda i,j,k,l: key[i,j,k,l] * te.rsqrt(key_sum[i,j,k,0] + eps),
            "k_norm"
        )
        # add intermediates to comp graph
        nodes.extend([query_sq, key_sq, query_sum, key_sum])
        query = q_norm
        key = k_norm

    query_transpose =

    basefunc: tir.PrimFunc = te.create_prim_func(nodes)
    return basefunc
    # TODO: once these functions actually work, experiment with loop fusions
    # etc. to make the kernels performant. May have to switch to hand-written
    # tir to really optimize it.

def _chunk_recurrent_gated_delta_rule():
    inputs = []
    outputs = []
    return te.create_prim_func(inputs + outputs)

print(_gen_causal_conv1d_update(128, 128).script())
print(_chunk_gated_delta_rule(128,128,128,128,use_qk_l2norm_in_kernel=False).script())
print(_chunk_gated_delta_rule(128,128,128,128,use_qk_l2norm_in_kernel=True).script())
