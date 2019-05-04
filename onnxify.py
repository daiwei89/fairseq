import onnx
import numpy as np
import torch
import os
import pickle
from os.path import dirname as dn
from fairseq import options, tasks, utils
from fairseq.tasks.translation import TranslationTask
from fairseq.data import Dictionary
from fairseq import tokenizer

repo_dir = dn(dn(dn(os.path.dirname(__file__))))

model_chpt = os.path.join(repo_dir, "model_zoo_tests", "nitro_model_zoo",
        "MT-bi-en_Var-zh_CN-v106-20190114-d299c44b0")
ckpt_path = os.path.join(model_chpt, 'checkpoint_best.pt')
dict_path = os.path.join(model_chpt, 'dict.en_zh.txt')

onnx_output_path = os.path.join(repo_dir, "onnx", \
        "MT-bi-en_Var-zh_CN-v106-20190114-d299c44b0")

def _prep_sample(bi_dict, example, pad=False, max_src_len=None):
    example_tokens = [bi_dict.index(x) for x in tokenizer.tokenize_line(example)]
    example_tokens.append(bi_dict.eos())
    src_len = torch.tensor([len(example_tokens)])
    if pad:
        example_tokens.extend([bi_dict.unk()]*(max_src_len - len(example_tokens)))
    src_seq = torch.from_numpy(np.array([example_tokens]))
    return src_seq, src_len

def load_transformer():
    # Load the MT Transformer model.
    ckpt = utils.load_checkpoint_to_cpu(ckpt_path)
    args = ckpt['args']
    state_dict = ckpt['model']
    bi_dict = Dictionary.load(dict_path)
    task = TranslationTask(args, bi_dict, bi_dict)
    model = task.build_model(args)
    model.upgrade_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.float()
    model.eval()
    model.encoder.eval()
    model.decoder.eval()
    return {"model": model, "bi_dict": bi_dict, "args": args}

def _greedy_translate(model, bi_dict, args, example):
    src_seq_flex, src_len_flex = _prep_sample(bi_dict, example)

    torch_in_flex_encoder = (src_seq_flex[:src_len_flex],
        src_len_flex, model.encoder.embed[:,:src_len_flex])
    torch_out_flex_encoder = model.encoder(*torch_in_flex_encoder)

    pos = 0
    max_target_pos = args.max_target_positions
    max_src_len = args.max_source_positions
    decoder_tokens = torch.empty(1, max_target_pos+1, dtype=torch.long).fill_(bi_dict.pad())
    decoder_tokens[:, 0] = bi_dict.eos()  # first decoding token is <eos>

    self_attn_kv = model.decoder.get_empty_self_att_cache(bsz=1,
            max_src_len=max_src_len)

    for pos in range(0, max_target_pos):
        torch_in_flex_decoder = (decoder_tokens[:, pos:(pos+1)],
                model.encoder.embed[:,pos], torch_out_flex_encoder[1],
                self_attn_kv[:,:,:,:,:pos], src_len_flex,
                torch.tensor(pos))
        torch_out_flex_decoder = model.decoder(*torch_in_flex_decoder)

        self_attn_kv = torch_out_flex_decoder[2]

        scores = torch_out_flex_decoder[0]
        best_token = scores.argmax().item()
        decoder_tokens[:, pos + 1] = best_token
        if best_token == bi_dict.eos():
            decoded_tokens = [bi_dict[i] for i in decoder_tokens[0, :pos+1].numpy()]
            print("test greedy generator\n", decoded_tokens)
            return decoded_tokens

def test_transformer_no_onnx():
    load_res = load_transformer()
    model, bi_dict, args = load_res["model"], load_res["bi_dict"], \
            load_res["args"]

    # Example data
    example = "I love you <src-en_US> <tar-zh_CN>"
    assert _greedy_translate(model, bi_dict, args, example) \
            == ['</s>', '我爱你']
    example = "hello world <src-en_US> <tar-zh_CN>"
    assert _greedy_translate(model, bi_dict, args, example) \
            == ['</s>', '你好', '世界']


def output_onnx(output_path):
    load_res = load_transformer()
    model, bi_dict, args = load_res["model"], load_res["bi_dict"], \
            load_res["args"]

    # Example data
    max_src_len = args.max_source_positions
    example = "hello world <src-en_US> <tar-zh_CN>"
    src_seq_encoder, src_len_encoder = _prep_sample(bi_dict, example,
            pad=True,
            max_src_len=max_src_len)

    # Ground truth output
    torch_in_encoder = (src_seq_encoder, src_len_encoder, model.encoder.embed)
    model.encoder.prepare_for_onnx_export_()
    torch_out_encoder = model.encoder.forward(*torch_in_encoder)

    # Onnxify
    model.decoder.prepare_for_onnx_export_()
    onnx_path = os.path.join(output_path, "encoder.onnx")
    torch.onnx._export(model.encoder, torch_in_encoder,
            onnx_path, verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            example_outputs=torch_out_encoder)

    # ====== Decoder ======

    decoder_token = torch.tensor([[bi_dict.eos()]]) # first token
    decoder_pos = 0

    max_target_pos = args.max_target_positions
    self_attn_kv = model.decoder.get_empty_self_att_cache(1, max_target_pos)
    torch_in_decoder = (decoder_token, model.encoder.embed[:,decoder_pos],
            torch_out_encoder[1], self_attn_kv, src_len_encoder,
            torch.tensor(decoder_pos))
    torch_out_decoder = model.decoder.forward(*torch_in_decoder)

    # Onnxify
    onnx_path = os.path.join(output_path, "decoder.onnx")
    torch.onnx._export(model.decoder, torch_in_decoder,
            onnx_path, verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            example_outputs=torch_out_decoder)

def export_embeddings(onnx_output_path):
    load_res = load_transformer()
    model = load_res["model"]
    embeddings = {
            "embed": model.encoder.embed.numpy(),
            "self_attn_kv":   \
                    model.decoder.get_empty_self_att_cache(1, 1024).numpy()
            }
    embedding_path = os.path.join(onnx_output_path, "embedding.pkl")
    with open(embedding_path, 'wb') as f:
        # For python2 compatibility.
        pickle.dump(embeddings, f, protocol=2)
    with open(embedding_path, 'rb') as f:
        b = pickle.load(f)
    assert np.array_equal(b["embed"], embeddings["embed"])


if __name__ == "__main__":
    test_transformer_no_onnx()
    if not os.path.exists(onnx_output_path):
        os.makedirs(onnx_output_path)
    export_embeddings(onnx_output_path)
    output_onnx(onnx_output_path)
    print("Onnx file exported to", onnx_output_path)
