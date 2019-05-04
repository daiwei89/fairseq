# Onnxified Fairseq-0.6.1

To onnxify Transformer to be run in
[NitroEspresso](https://gitlab-turi.corp.apple.com/turi/nitro_converter), you need:

1. A Python3 environment (e.g., virtualenv)
2. Have read access to Blobby `s3://nitro_model_zoo` bucket to download the trained weights

Run the followings in the root directory of the repo:

```
pip install -r requirements.txt
sh download_models.sh
python onnxify.py
```

The resulting onnx files are under
`onnx/MT-bi-en_Var-zh_CN-v106-20190114-d299c44b0/`

Note: This version was forked from Fairseq repo on 09 Feb, 2019, corresponding
to [this
commit](https://github.com/pytorch/fairseq/tree/fbd4cef9a575b5f77ca05d4b7c3ad3adb11141ac)

## Behind the Scene:

It's highly nontrivial to convert Fairseq Transformer to onnx and ensure proper
caching of decoder states for incremental decoding. Some major changes are:

1. Since `incremental_state` in Fairseq `TransformerDecoder` is a Python
   dictionary (not supported in PyTorch JIT until v1.1.0), we need to convert
   `incremental_state`, a messy nested dictionaries, to two 6-D tensors:

   `encoder_kv: [num_layers, 2, batch_size, num_heads, max_source_positions, head_dim]`

   `self_attn_kv: [num_layers, 2, batch_size, num_heads, max_target_positions, head_dim]`

   where usually `num_layers=6, batch_size=1, num_heads=8,
   max_*_positions=1024, head_dim=64`. 2 is for `{key, value}` of the
   attention.

2. Properly output all incremental state. In the original Fairseq the
   `encoder_kv` states are computed once during the first decoding step and is
   cached. Since PyTorch JIT graph doesn't support `if` statement, this needs
   two decoder computation graphs to achieve: the first graph computes the
   `encoder_kv` and decode the first token, and the second graph decode the
   rest of decoding tokens and skip computing `encoder_kv`. Not only is this
   cumbersome, it stores the decoder weights twice (in two ONNX graphs).

   The solution is to move the computation of `encoder_kv` from the decoder to
   encoder. Thus the encoder outputs the `encoder_kv` for each layer of
   decoder. Our encoder still outputs the original encoded representation, but
   it's not used. When we package into Nitro program, the encoder will contain
   projection weights to generate `encoder_kv`, and decoder will not have
   those weights.

   The decoder also needs to properly take in the self attention KV caches
   (`self_attn_kv`) and return the updated cache.

3. Because PyTorch `jit.trace` does not support any dynamic tensor shape (and Torch
   Script doesn't work with `nn.ModuleList` at the time of this writing), to enable
   dynamic computation in NitroEspresso we need to do a few technical tricks
   during the conversion process so we don't have to fork PyTorch itself (much
   more challenging).

4. Similar to #3, because `jit.trace` doesn't record any tensor shape
   dynamically (shape changing with the input), positional embedding
   computation is impossible. The workaround is to pre-compute the embedding
   and output as part of `embedding.pkl` and eliminate the entire dynamic
   embedding computation.


Despite the changes above, we use the identical weights as `fairseq-0.6.1`.
This means that if you train the model using the original `fairseq-0.6.1`, it
can be loaded and onnxified using the above process, and the onnx can be
further converted to Nitro and executed in Espresso. The code has been tested
to reach numerical parity with the original fairseq-0.6.1 on a few samples.

